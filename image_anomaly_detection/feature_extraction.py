"""
Provides classes and functions for extracting embedding vectors
"""

import torch
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm



class ResnetFeaturesExtractor(torch.nn.Module):

    def __init__(self, backbone_name, device):
        super().__init__()

        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True, progress=True)
        elif backbone_name == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(pretrained=True, progress=True)

        self.backbone.to(device)
        self.backbone.eval()
        self.eval()


    def to(self, device=None, dtype=None, non_blocking=False):
        self.backbone.to(device, dtype=dtype, non_blocking=non_blocking)


    def forward(self, batch, channel_indices=None, layer_hook=None, layer_indices=None):

        with torch.no_grad():

            batch = self.backbone.conv1(batch)
            batch = self.backbone.bn1(batch)
            batch = self.backbone.relu(batch)
            batch = self.backbone.maxpool(batch)

            layer1 = self.backbone.layer1(batch)
            layer2 = self.backbone.layer2(layer1)
            layer3 = self.backbone.layer3(layer2)
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


    def from_dataloader(self, dataloader, channel_indices=None,
                        layer_hook=None, layer_indices=None):

        embedding_vectors = None

        for (batch, _, _) in tqdm(dataloader, 'Feature extraction'):

            batch_embedding_vectors = self(batch,
                                            channel_indices=channel_indices,
                                            layer_hook=layer_hook,
                                            layer_indices=layer_indices)

            if embedding_vectors is None:
                embedding_vectors = batch_embedding_vectors
            else:
                embedding_vectors = torch.cat((embedding_vectors, batch_embedding_vectors), 0)

        return embedding_vectors



def concatenate_layers(layers):
    expanded_layers = layers[0]
    for layer in layers[1:]:
        expanded_layers = concatenate_two_layers(expanded_layers, layer)
    return expanded_layers



def concatenate_two_layers(layer1, layer2):
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
