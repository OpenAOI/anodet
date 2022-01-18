"""
Provides classes and functions for working with PatchCore.
"""

import math
import torch
import numpy as np
import cv2
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from typing import Optional, Callable, List, Tuple
from tqdm import tqdm
from .sampling_methods.kcenter_greedy import kCenterGreedy
from .feature_extraction import ResnetEmbeddingsExtractor

class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = torch.cdist(x, self.train_pts, self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]

class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        dist = torch.cdist(x, self.train_pts, self.p)
        knn = dist.topk(self.k, largest=False)
        return knn


class PatchCore:
    """A PatchCore model with functions to train and perform inference."""

    def __init__(self, backbone: str = 'wide_resnet50',
                 device: torch.device = torch.device('cpu'),
                 embedding_coreset: Optional[torch.Tensor] = None,
                 channel_indices: Optional[torch.Tensor] = None,
                 layer_indices: Optional[List[int]] = None,
                 layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:

        """Construct the model and initialize the attributes

        Args:
            backbone: The name of the desired backbone. Must be one of: [resnet18, wide_resnet50].
            device: The device where to run the model.
            embedding_coreset: A tensor with the coreset, with size (N, D), \
                where N is the number of vectors and D is the number of channel_indices.
            channel_indices: A tensor with the desired channel indices to extract \
                from the backbone, with size (D).
            layer_indices: A list with the desired layers to extract from the backbone, \
            allowed indices are 1, 2, 3 and 4.
            layer_hook: A function that can modify the layers during extraction.
        """

        self.device = device
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        self.embedding_coreset = embedding_coreset
        self.channel_indices = channel_indices

        self.layer_indices = layer_indices
        if self.layer_indices is None:
            self.layer_indices = [1, 2]

        self.layer_hook = layer_hook
        if self.layer_hook is None:
            self.layer_hook = torch.nn.AvgPool2d(3, 1, 1)

        self.to_device(self.device)

    def to_device(self, device: torch.device) -> None:
        """Perform device conversion on backone, mean, cov_inv and channel_indices

        Args:
            device: The device where to run the model.

        """

        self.device = device
        if self.embeddings_extractor is not None:
            self.embeddings_extractor.to_device(device)
        if self.channel_indices is not None:
            self.channel_indices = self.channel_indices.to(device)

    def fit(self, dataloader: torch.utils.data.DataLoader,
            sampling_ratio: float = 0.001) -> None:

        """Fit the model (i.e. embedding_coreset) to data.

        Args:
            dataloader: A pytorch dataloader, with sample dimensions (B, D, H, W), \
                containing normal images.

        """

        embedding_vectors = self.embeddings_extractor.from_dataloader(
            dataloader,
            channel_indices=self.channel_indices,
            layer_hook=self.layer_hook,
            layer_indices=self.layer_indices
        )

        batch_length, vector_num, channel_num = embedding_vectors.shape
        embedding_vectors = embedding_vectors.reshape(batch_length*vector_num,
                                                      channel_num).cpu().numpy()

        randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
        randomprojector.fit(embedding_vectors)
        # Coreset subsampling
        selector = kCenterGreedy(embedding_vectors, 0, 0)
        selected_idx = selector.select_batch(model=randomprojector, already_selected=[],
                                             N=int(embedding_vectors.shape[0]*sampling_ratio))

        self.embedding_coreset = embedding_vectors[selected_idx]
        print('initial embedding size : ', embedding_vectors.shape)
        print('final embedding size : ', self.embedding_coreset.shape)

    def predict(self,
                batch: torch.Tensor,
                n_neighbors: int = 9,
                apply_gaussian: bool = True,
                apply_resize: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        """Make a prediction on test images.

        Args:
            batch: A batch of test images, with dimension (B, D, h, w).
            n_neighbors: See documentation of sklearn.neighbors.NearestNeighbors.
            nn_metric: See documentation of sklearn.neighbors.NearestNeighbors.
            nn_metric: See documentation of sklearn.neighbors.NearestNeighbors.
            apply_gaussian: If true apply gaussian blur on score map.
            apply_resize: If true resize the score_map to size of images in batch.

        Returns:
            image_scores: A tensor with the image level scores, with dimension (B).
            score_map: A tensor with the patch level scores, with dimension (B, H, W)

        """

        assert self.embedding_coreset is not None, \
            "The model must be fitted or provided with embedding_coreset"

        embedding_vectors = self.embeddings_extractor(batch,
                                                      channel_indices=self.channel_indices,
                                                      layer_hook=self.layer_hook,
                                                      layer_indices=self.layer_indices
                                                      )

        knn = KNN(torch.from_numpy(self.embedding_coreset).to(str(self.device)), k=n_neighbors)
        patch_width = int(math.sqrt(embedding_vectors.shape[1]))
        score_maps = torch.zeros((embedding_vectors.shape[0], batch.shape[2], batch.shape[2]))

        image_scores = torch.zeros(embedding_vectors.shape[0])

        for i in range(embedding_vectors.shape[0]):
            patch_score = knn(embedding_vectors[i])[0].cpu().detach().numpy()
            score_map = patch_score[:, 0].reshape((patch_width, patch_width))

            N_b = patch_score[np.argmax(patch_score[:, 0])]
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            image_scores[i] = w*max(patch_score[:, 0])

            if apply_resize:
                score_map = cv2.resize(score_map, (batch.shape[2], batch.shape[2]))
            if apply_gaussian:
                score_map = torch.from_numpy(gaussian_filter(score_map, sigma=4))
            score_maps[i] = score_map

        return image_scores, score_maps

    def evaluate(self,
                 dataloader: torch.utils.data.DataLoader,
                 n_neighbors: int = 9,
                 nn_algorithm: str = "ball_tree",
                 nn_metric: str = "minkowski",
                 apply_gaussian: bool = True,
                 apply_resize: bool = True
                 ):

        """Run predict on all images in a dataloader and return the results.

        Args:
            dataloader: A pytorch dataloader, with sample dimensions (B, D, H, W), \
                containing normal images.
            n_neighbors: See documentation of sklearn.neighbors.NearestNeighbors.
            nn_metric: See documentation of sklearn.neighbors.NearestNeighbors.
            nn_metric: See documentation of sklearn.neighbors.NearestNeighbors.
            apply_gaussian: If true apply gaussian blur on score map.
            apply_resize: If true resize the score_map to size of images in batch.

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

        for (batch, image_classifications, masks) in tqdm(dataloader, 'Inference'):
            batch_image_scores, batch_score_maps = \
                self.predict(batch,
                             n_neighbors,
                             nn_algorithm,
                             nn_metric,
                             apply_gaussian,
                             apply_resize
                             )

            images.extend(batch.cpu().numpy())
            image_classifications_target.extend(image_classifications.cpu().numpy())
            masks_target.extend(masks.cpu().numpy())
            image_scores.extend(batch_image_scores.cpu().numpy())
            score_maps.extend(batch_score_maps.cpu().numpy())

        return np.array(images), np.array(image_classifications_target), \
            np.array(masks_target).flatten().astype(np.uint8), \
            np.array(image_scores), np.array(score_maps).flatten()
