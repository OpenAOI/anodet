"""
Provides classes and functions for extracting embedding vectors from neural networks.
"""

import torch
import torchinfo
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, wide_resnet50_2, Wide_ResNet50_2_Weights
from tqdm import tqdm
from typing import List, Optional, Callable, cast
from torch.utils.data import DataLoader


class ResnetEmbeddingsExtractor(torch.nn.Module):
    """A class to hold, and extract embedding vectors from, a resnet.

    Attributes:
        backbone: The resnet from which to extract embedding vectors.

    """

    def __init__(self, backbone_name: str, device: torch.device) -> None:
        """Construct the backbone and set appropriate mode and device

        Args:
            backbone_name: The name of the desired backbone. Must be
                one of: [resnet18, wide_resnet50].
            device: The device where to run the network.

        """

        super().__init__()
        assert backbone_name in ['resnet18', 'wide_resnet50']

        if backbone_name == 'resnet18':
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT, progress=True)
        elif backbone_name == 'wide_resnet50':
            self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT, progress=True)

        self.backbone.to(device)
        self.backbone.eval()
        self.eval()

    def to_device(self, device: torch.device) -> None:
        """Perform device conversion on backone

        See pytorch docs for documentation on torch.Tensor.to

        """
        self.backbone.to(device)

    @torch.no_grad() # Cannot backpropagate but improves performance
    def forward(self,
                batch: torch.Tensor,
                channel_indices: Optional[torch.Tensor] = None,
                layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                layer_indices: Optional[List[int]] = None
                ) -> torch.Tensor:
        """Run inference on backbone and return the embedding vectors.

        Args:
            batch: A batch of images.
            channel_indices: A list of indices with the desired channels to include in
                the embedding vectors.
            layer_hook: A function that runs on each layer of the resnet before
                concatenating them.
            layer_indices: A list of indices with the desired layers to include in the
                embedding vectors.

        Returns:
            embedding_vectors: The embedding vectors.

        """

        # Initial convolutional layer
        batch = self.backbone.conv1(batch)  

        # Batch normalization (BatchNorm2d)
        batch = self.backbone.bn1(batch)  

        # Rectified Linear Unit (ReLU) activation function
        batch = self.backbone.relu(batch)  

        # Downsampling to reduce spatial dimensions and retain important features
        batch = self.backbone.maxpool(batch)  

        # Capture low-level features and spatial patterns
        layer1 = self.backbone.layer1(batch)  

        # Build upon layer1, capturing more complex features
        layer2 = self.backbone.layer2(layer1)  

        # Capture higher-level features and more abstract representations
        layer3 = self.backbone.layer3(layer2)  

        # Refine features, preparing for final classification
        layer4 = self.backbone.layer4(layer3)

        layers = [layer1, layer2, layer3, layer4]  

        if layer_indices is not None:
            layers = [layers[i] for i in layer_indices]

        if layer_hook is not None:
            layers = [layer_hook(layer) for layer in layers]

        embedding_vectors = concatenate_layers(layers)

        if channel_indices is not None:
            embedding_vectors = torch.index_select(embedding_vectors, 1, channel_indices)

        batch_size, length, width, height = embedding_vectors.shape
        embedding_vectors = embedding_vectors.reshape(batch_size, length, width*height)
        embedding_vectors = embedding_vectors.permute(0, 2, 1)

        return embedding_vectors

    def from_dataloader(self,
                        dataloader: DataLoader,
                        channel_indices: Optional[torch.Tensor] = None,
                        layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                        layer_indices: Optional[List[int]] = None
                        ) -> torch.Tensor:
        """Same as self.forward but take a dataloader instead of a tensor as argument."""

        embedding_vectors: Optional[torch.Tensor] = None

        for (batch, _, _) in tqdm(dataloader, 'Feature extraction'):

            batch_embedding_vectors = self(batch,
                                           channel_indices=channel_indices,
                                           layer_hook=layer_hook,
                                           layer_indices=layer_indices)

            if embedding_vectors is None:
                embedding_vectors = batch_embedding_vectors
            else:
                embedding_vectors = torch.cat((embedding_vectors, batch_embedding_vectors), 0)

        return cast(torch.Tensor, embedding_vectors)

    def get_summary(self, input_size = (1, 3, 224, 224)):
        """Get a summary of the backbone model structure."""
        
        # input_size with standard_image_transform
        return torchinfo.summary(self.backbone, input_size)



def concatenate_layers(layers: List[torch.Tensor]) -> torch.Tensor:
    """Scale all tensors to the heigth and width of the first tensor and concatenate them."""

    expanded_layers = layers[0]
    for layer in layers[1:]:
        expanded_layers = concatenate_two_layers(expanded_layers, layer)
    return expanded_layers


def concatenate_two_layers(layer1: torch.Tensor, layer2: torch.Tensor) -> torch.Tensor:
    """Scale the second tensor to the height and width of the first tensor and concatenate them."""

    device = layer1.device
    batch_length, channel_num1, height1, width1 = layer1.size()
    _, channel_num2, height2, width2 = layer2.size()
    height_ratio = int(height1 / height2)
    layer1 = F.unfold(layer1, kernel_size=height_ratio, dilation=1, stride=height_ratio)
    layer1 = layer1.view(batch_length, channel_num1, -1, height2, width2)
    result = torch.zeros(batch_length, channel_num1 + channel_num2, layer1.size(2),
                         height2, width2, device=device)
    for i in range(layer1.size(2)):
        result[:, :, i, :, :] = torch.cat((layer1[:, :, i, :, :], layer2), 1)
    del layer1
    del layer2
    result = result.view(batch_length, -1, height2 * width2)
    result = F.fold(result, kernel_size=height_ratio,
                    output_size=(height1, width1), stride=height_ratio)
    return result

if __name__ == "__main__":
    nn_resnet_18 = ResnetEmbeddingsExtractor(backbone_name="resnet18", device='cpu')
    nn_resnet_18.get_summary()


    nn_resnet_50 = ResnetEmbeddingsExtractor(backbone_name="wide_resnet50", device='cpu')
    nn_resnet_50.get_summary()


