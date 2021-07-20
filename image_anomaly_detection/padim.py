"""
Provides classes and functions for working with PaDiM.
"""

import math
import random
from typing import Optional, Callable, List, Tuple
import torch
from torchvision import transforms as T
import torch.nn.functional as F
from .feature_extraction import ResnetEmbeddingsExtractor
from .utils import pytorch_cov, mahalanobis


class Padim:
    """A padim model with functions to train and perform inference."""

    def __init__(self, backbone: str = 'resnet18',
                 device: torch.device = torch.device('cpu'),
                 mean: Optional[torch.Tensor] = None,
                 cov_inv: Optional[torch.Tensor] = None,
                 channel_indices: Optional[torch.Tensor] = None,
                 layer_indices: Optional[List[int]] = None,
                 layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:

        self.device = device
        self.embeddings_extractor = ResnetEmbeddingsExtractor(backbone, self.device)
        self.mean = mean
        self.cov_inv = cov_inv

        self.channel_indices = channel_indices
        if self.channel_indices is None:
            if backbone == 'resnet18':
                self.channel_indices = get_indices(100, 448, self.device)
            elif backbone == 'wide_resnet50':
                self.channel_indices = get_indices(550, 1792, self.device)

        self.layer_indices = layer_indices
        if self.layer_indices is None:
            self.layer_indices = [0, 1, 2]

        self.layer_hook = layer_hook
        self.to_device(self.device)

    def to_device(self, device: torch.device) -> None:
        self.device = device
        if self.embeddings_extractor is not None:
            self.embeddings_extractor.to_device(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.cov_inv is not None:
            self.cov_inv = self.cov_inv.to(device)
        if self.channel_indices is not None:
            self.channel_indices = self.channel_indices.to(device)

    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:

        embedding_vectors = self.embeddings_extractor.from_dataloader(
            dataloader,
            channel_indices=self.channel_indices,
            layer_hook=self.layer_hook,
            layer_indices=self.layer_indices
        )

        self.mean = torch.mean(embedding_vectors, dim=0)
        cov = pytorch_cov(embedding_vectors.permute(1, 0, 2), rowvar=False) \
            + 0.01 * torch.eye(embedding_vectors.shape[2])
        self.cov_inv = torch.inverse(cov)

    def predict(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.mean is not None and self.cov_inv is not None

        embedding_vectors = self.embeddings_extractor(batch,
                                                      channel_indices=self.channel_indices,
                                                      layer_hook=self.layer_hook,
                                                      layer_indices=self.layer_indices
                                                      )

        patch_scores = mahalanobis(self.mean, self.cov_inv, embedding_vectors)

        patch_width = int(math.sqrt(embedding_vectors.shape[1]))
        patch_scores = patch_scores.reshape(batch.shape[0], patch_width, patch_width)

        score_map = F.interpolate(patch_scores.unsqueeze(1), size=batch.shape[2],
                                  mode='bilinear', align_corners=False).squeeze()
        if batch.shape[0] == 1:
            score_map = score_map.unsqueeze(0)
        score_map = T.GaussianBlur(33, sigma=4)(score_map)

        image_scores = torch.max(score_map.reshape(score_map.shape[0], -1), -1).values

        return image_scores, score_map


def get_indices(choose, total, device):
    random.seed(1024)
    torch.manual_seed(1024)

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(1024)

    return torch.tensor(random.sample(range(0, total), choose), device=device)
