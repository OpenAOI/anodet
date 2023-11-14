"""
Provides classes and functions for working with PaDiM.
"""

import math
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from tqdm import tqdm

from .feature_extraction import ResnetEmbeddingsExtractor
from .utils import mahalanobis, pytorch_cov, split_tensor_and_run_function


class Padim:
    """A padim model with functions to train and perform inference."""

    def __init__(
        self,
        backbone: str = "resnet18",
        device: torch.device = torch.device("cpu"),
        mean: Optional[torch.Tensor] = None,
        cov_inv: Optional[torch.Tensor] = None,
        channel_indices: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
        layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """Construct the model and initialize the attributes

        Args:
            backbone: The name of the desired backbone. Must be one of: [resnet18, wide_resnet50].
            device: The device where to run the model.
            mean: A tensor with the mean vectors of each patch with size (D, H, W), \
                where D is the number of channel_indices.
            cov_inv: A tensor with the inverse of the covariance matrices of each patch \
                with size (D, D, H, W), where D is the number of channel_indices.
            channel_indices: A tensor with the desired channel indices to extract \
                from the backbone, with size (D).
            layer_indices: A list with the desired layers to extract from the backbone, \
            allowed indices are 1, 2, 3 and 4.
            layer_hook: A function that can modify the layers during extraction.
        """
        self.backbone = backbone
        self.device = device
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        self.mean = mean
        self.cov_inv = cov_inv

        # Embeddings/feature extraction done in feature_extraction.py (kernel).
        # The kernel will try to extract the important features of the images and remove
        # the background noise.

        # Channels will store this extracted information which will act as an input for the
        # next kernel (or layer). Some channels contain gradients and color shades, others
        # contain sharp edges found the images.

        self.channel_indices = channel_indices
        if self.channel_indices is None:
            if backbone == "resnet18":
                self.channel_indices = get_indices(100, 448, self.device)
            elif backbone == "wide_resnet50":
                self.channel_indices = get_indices(550, 1792, self.device)

        # List that determines which layers to extract from the backbone.
        # Each layer captures different blocks of the backbone network.
        self.layer_indices = layer_indices
        if self.layer_indices is None:
            self.layer_indices = [0, 1, 2]

        # Layer hook performs operations on all layers, as of now, no layer hooks are applied
        self.layer_hook = layer_hook
        self.to_device(self.device)

    def to_device(self, device: torch.device) -> None:
        """Perform device conversion on backone, mean, cov_inv and channel_indices

        Args:
            device: The device where to run the model.

        """

        self.device = device
        if self.embeddings_extractor is not None:
            self.embeddings_extractor.to_device(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.cov_inv is not None:
            self.cov_inv = self.cov_inv.to(device)
        if self.channel_indices is not None:
            self.channel_indices = self.channel_indices.to(device)

    def fit(
        self, dataloader: torch.utils.data.DataLoader, extractions: int = 1
    ) -> None:
        """Fit the model (i.e. mean and cov_inv) to data.

        Args:
            dataloader: A pytorch dataloader, with sample dimensions (B, D, H, W), \
                containing normal images. (Batch size, depth (channel), height, width)
            extractions: Number of extractions from dataloader. Could be of interest \
                when applying random augmentations.

        """
        embedding_vectors = None
        for i in range(extractions):
            extracted_embedding_vectors = self.embeddings_extractor.from_dataloader(
                dataloader,
                channel_indices=self.channel_indices,
                layer_hook=self.layer_hook,
                layer_indices=self.layer_indices,
            )
            if embedding_vectors is None:
                embedding_vectors = extracted_embedding_vectors
            else:
                embedding_vectors = torch.cat(
                    (embedding_vectors, extracted_embedding_vectors), 0
                )

        self.mean = torch.mean(embedding_vectors, dim=0)
        cov = pytorch_cov(
            embedding_vectors.permute(1, 0, 2), rowvar=False
        ) + 0.01 * torch.eye(embedding_vectors.shape[2])
        # Run inverse function on splitted tensor to save ram memory
        self.cov_inv = split_tensor_and_run_function(
            func=torch.inverse, tensor=cov, split_size=1
        )

    def predict(
        self, batch: torch.Tensor, gaussian_blur: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make a prediction on test images.

        Args:
            batch: A batch of test images, with dimension (B, D, h, w).

        Returns:
            image_scores: A tensor with the image level scores, with dimension (B).
            score_map: A tensor with the patch level scores, with dimension (B, H, W)

        """

        assert (
            self.mean is not None and self.cov_inv is not None
        ), "The model must be trained or provided with mean and cov_inv"

        embedding_vectors = self.embeddings_extractor(
            batch,
            channel_indices=self.channel_indices,
            layer_hook=self.layer_hook,
            layer_indices=self.layer_indices,
        )

        patch_scores = mahalanobis(self.mean, self.cov_inv, embedding_vectors)

        patch_width = int(math.sqrt(embedding_vectors.shape[1]))
        patch_scores = patch_scores.reshape(batch.shape[0], patch_width, patch_width)

        score_map = F.interpolate(
            patch_scores.unsqueeze(1),
            size=batch.shape[2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        if batch.shape[0] == 1:
            score_map = score_map.unsqueeze(0)

        if gaussian_blur:
            score_map = T.GaussianBlur(33, sigma=4)(score_map)

        image_scores = torch.max(score_map.reshape(score_map.shape[0], -1), -1).values

        return image_scores, score_map

    def evaluate(
        self, dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run predict on all images in a dataloader and return the results.

        Args:
            dataloader: A pytorch dataloader, with sample dimensions (B, D, H, W), \
                containing normal images.

        Returns:
            images: An array containing all input images.
            image_classifications_target: An array containing the target \
                classifications on image level.
            masks_target: An array containing the target classifications on patch level.
            image_scores: An array containing the predicted scores on image level.
            score_maps: An array containing the predicted scores on patch level.

        """

        images = []
        image_classifications_target = []
        masks_target = []
        image_scores = []
        score_maps = []

        for batch, image_classifications, masks in tqdm(dataloader, "Inference"):
            batch_image_scores, batch_score_maps = self.predict(batch)

            images.extend(batch.cpu().numpy())
            image_classifications_target.extend(image_classifications.cpu().numpy())
            masks_target.extend(masks.cpu().numpy())
            image_scores.extend(batch_image_scores.cpu().numpy())
            score_maps.extend(batch_score_maps.cpu().numpy())

        return (
            np.array(images),
            np.array(image_classifications_target),
            np.array(masks_target).flatten().astype(np.uint8),
            np.array(image_scores),
            np.array(score_maps).flatten(),
        )

    def __repr__(self) -> str:
        attributes = [f"backbone={self.backbone if self.backbone else 'None'}",
                      f"device={self.device}",
                      f"mean={self.mean if self.mean is not None else 'None'}",
                      f"cov_inv={self.cov_inv if self.cov_inv is not None else 'None'}",
                      f"channel_indices={self.channel_indices if self.channel_indices is not None else 'None'}",
                      f"layer_indices={self.layer_indices if self.layer_indices is not None else 'None'}",
                      f"layer_hook={self.layer_hook if self.layer_hook is not None else 'None'}"]
        return f"Padim({', '.join(attributes)})"


def get_indices(choose, total, device):
    random.seed(1024)
    torch.manual_seed(1024)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(1024)

    return torch.tensor(random.sample(range(0, total), choose), device=device)