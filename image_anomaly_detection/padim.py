import math
import random
from typing import Optional, Callable, List
import torch
from torchvision import transforms as T
import numpy as np
from .feature_extraction import ResnetFeaturesExtractor
from .utils import to_batch, pytorch_cov, mahalanobis



class Padim:

    def __init__(self, device: torch.device, backbone_name: str,
                 mean: Optional[torch.Tensor] = None,
                 cov_inv: Optional[torch.Tensor] = None,
                 transform: Optional[T.Compose] = None,
                 channel_indices: Optional[torch.Tensor] = None,
                 layer_indices: Optional[List[int]] = None,
                 layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:

        self.device = device
        self.features_extractor = ResnetFeaturesExtractor(backbone_name, self.device)

        self.transform = transform
        if self.transform is None:
            self.transform = T.Compose([T.Resize(224),
                                        T.CenterCrop(224),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                       ])

        self.mean = mean
        self.cov_inv = cov_inv

        self.channel_indices = channel_indices
        if self.channel_indices is None:
            if backbone_name == 'resnet18':
                self.channel_indices = get_indices(100, 448, self.device)
            elif backbone_name == 'wide_resnet50':
                self.channel_indices = get_indices(550, 1792, self.device)

        self.layer_indices = layer_indices
        if self.layer_indices is None:
            self.layer_indices = [0,1,2]

        self.layer_hook = layer_hook

        self.to_device(self.device)


    def to_device(self, device: torch.device) -> None:
        self.device = device
        if self.features_extractor is not None:
            self.features_extractor.to(device)
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.cov_inv is not None:
            self.cov_inv = self.cov_inv.to(device)
        if self.channel_indices is not None:
            self.channel_indices = self.channel_indices.to(device)


    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:

        embedding_vectors = self.features_extractor.from_dataloader(
            dataloader,
            channel_indices=self.channel_indices,
            layer_hook=self.layer_hook,
            layer_indices=self.layer_indices
        )

        self.mean = torch.mean(embedding_vectors, dim=0)
        cov = pytorch_cov(embedding_vectors.permute(1, 0, 2), rowvar=False) \
        + 0.01 * torch.eye(embedding_vectors.shape[2])
        self.cov_inv = torch.inverse(cov)


    def predict(self, images: List[np.ndarray]) -> torch.Tensor:
        assert self.mean is not None and self.cov_inv is not None

        batch = to_batch(images, self.transform, self.device)
        embedding_vectors = self.features_extractor(batch,
                                                    channel_indices=self.channel_indices,
                                                    layer_hook=self.layer_hook,
                                                    layer_indices=self.layer_indices)

        patch_scores = mahalanobis(self.mean, self.cov_inv, embedding_vectors)

        patch_width = int(math.sqrt(embedding_vectors.shape[1]))
        patch_scores = patch_scores.reshape(batch.shape[0], patch_width, patch_width)

        return patch_scores





def get_original_resnet18_indices(device):
    return get_indices(100, 448, device)



def get_original_wide_resnet50_indices(device):
    return get_indices(550, 1792, device)



def get_indices(choose, total, device):
    random.seed(1024)
    torch.manual_seed(1024)

    if device.type == 'cuda':
        torch.cuda.manual_seed_all(1024)

    return torch.tensor(random.sample(range(0, total), choose), device=device)
