import math
from typing import Optional, Callable, List, Tuple
import torch
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from .sampling_methods.kcenter_greedy import kCenterGreedy
from .feature_extraction import ResnetFeaturesExtractor
from scipy.ndimage import gaussian_filter
import cv2


class PatchCore:

    def __init__(self, backbone: str = 'resnet18',
                 device: torch.device = torch.device('cpu'),
                 embedding_coreset: Optional[torch.Tensor] = None,
                 channel_indices: Optional[torch.Tensor] = None,
                 layer_indices: Optional[List[int]] = None,
                 layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:

        self.device = device
        self.features_extractor = ResnetFeaturesExtractor(backbone, self.device)

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
        self.device = device
        if self.features_extractor is not None:
            self.features_extractor.to(device)
        if self.channel_indices is not None:
            self.channel_indices = self.channel_indices.to(device)

    def fit(self, dataloader: torch.utils.data.DataLoader,
            sampling_ratio: float = 0.001) -> None:

        embedding_vectors = self.features_extractor.from_dataloader(
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

        selector = kCenterGreedy(embedding_vectors, 0, 0)
        selected_idx = selector.select_batch(model=randomprojector, already_selected=[],
                                             N=int(embedding_vectors.shape[0]*sampling_ratio))

        self.embedding_coreset = embedding_vectors[selected_idx]

    def predict(self, batch: torch.Tensor,
                n_neighbors: int = 9) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.embedding_coreset is not None

        embedding_vectors = self.features_extractor(batch,
                                                    channel_indices=self.channel_indices,
                                                    layer_hook=self.layer_hook,
                                                    layer_indices=self.layer_indices)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree',
                                metric='minkowski', p=2).fit(self.embedding_coreset)

        patch_width = int(math.sqrt(embedding_vectors.shape[1]))
        score_maps = torch.zeros((embedding_vectors.shape[0], batch.shape[2], batch.shape[2]))

        image_scores = torch.zeros(embedding_vectors.shape[0])

        for i in range(embedding_vectors.shape[0]):
            patch_score, _ = nbrs.kneighbors(embedding_vectors[i].cpu().numpy())
            score_map = patch_score[:, 0].reshape((patch_width, patch_width))

            N_b = patch_score[np.argmax(patch_score[:, 0])]
            w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
            image_scores[i] = w*max(patch_score[:, 0])

            score_map = cv2.resize(score_map, (batch.shape[2], batch.shape[2]))
            score_map = torch.from_numpy(gaussian_filter(score_map, sigma=4))
            score_maps[i] = score_map

        return image_scores, score_maps
