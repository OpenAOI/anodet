import math
from typing import Optional, Callable, List
import torch
from torchvision import transforms as T
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from .sampling_methods.kcenter_greedy import kCenterGreedy
from .feature_extraction import ResnetFeaturesExtractor
from .utils import to_batch



class PatchCore:

    def __init__(self, backbone_name: str,
                 embedding_coreset: Optional[torch.Tensor] = None,
                 device: Optional[torch.device] = None,
                 transforms: Optional[T.Compose] = None,
                 channel_indices: Optional[torch.Tensor] = None,
                 layer_indices: Optional[List[int]] = None,
                 layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:

        self.device = device
        if self.device is None:
            self.device = torch.device('cpu')

        self.features_extractor = ResnetFeaturesExtractor(backbone_name, self.device)

        self.transforms = transforms
        if self.transforms is None:
            self.transforms = T.Compose([T.Resize(224),
                                        T.CenterCrop(224),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                       ])


        self.embedding_coreset = embedding_coreset

        self.channel_indices = channel_indices


        self.layer_indices = layer_indices
        if self.layer_indices is None:
            self.layer_indices = [1,2]

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



    def predict(self, images: List[np.ndarray], n_neighbors: int = 9) -> torch.Tensor:
        assert self.embedding_coreset is not None

        batch = to_batch(images, self.transforms, self.device)
        embedding_vectors = self.features_extractor(batch,
                                                    channel_indices=self.channel_indices,
                                                    layer_hook=self.layer_hook,
                                                    layer_indices=self.layer_indices)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree',
                                metric='minkowski', p=2).fit(self.embedding_coreset)

        patch_width = int(math.sqrt(embedding_vectors.shape[1]))
        patch_scores = torch.zeros((embedding_vectors.shape[0], patch_width, patch_width))

        for i in range(embedding_vectors.shape[0]):
            patch_score, _ = nbrs.kneighbors(embedding_vectors[i].cpu().numpy())
            patch_scores[i] = torch.from_numpy(patch_score[:,0].reshape((patch_width,patch_width)))

        return patch_scores
