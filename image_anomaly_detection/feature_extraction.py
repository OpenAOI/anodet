"""
Provides functions for extracting embedding vectors
"""

import random
import torch
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm



class WideResnet50Features(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.resnet50_2 = models.wide_resnet50_2(pretrained=True, progress=True)
        self.resnet50_2.to(device)
        self.resnet50_2.eval()

    def forward(self, batch):
        batch = self.resnet50_2.conv1(batch)
        batch = self.resnet50_2.bn1(batch)
        batch = self.resnet50_2.relu(batch)
        prelayer_out = self.resnet50_2.maxpool(batch)

        layer1 = self.resnet50_2.layer1(prelayer_out)
        layer2 = self.resnet50_2.layer2(layer1)
        layer3 = self.resnet50_2.layer3(layer2)
        #layer4_out = self.resnet18.layer4(layer3)

        return layer1, layer2, layer3



class Resnet18Features(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True, progress=True)
        self.resnet18.to(device)
        self.resnet18.eval()

    def forward(self, batch):
        batch = self.resnet18.conv1(batch)
        batch = self.resnet18.bn1(batch)
        batch = self.resnet18.relu(batch)
        prelayer_out = self.resnet18.maxpool(batch)

        layer1 = self.resnet18.layer1(prelayer_out)
        layer2 = self.resnet18.layer2(layer1)
        layer3 = self.resnet18.layer3(layer2)
        #layer4_out = self.resnet18.layer4(layer3)

        return layer1, layer2, layer3



def concatenate_layers(layers, device):
    expanded_layers = layers[0]
    for layer in layers[1:]:
        expanded_layers = concatenate_two_layers(expanded_layers, layer, device)
    return expanded_layers



def concatenate_two_layers(layer1, layer2, device):
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



def extract_embedding_vectors(model, batch, device, indices=None):
    with torch.no_grad():
        batch = batch.to(device)
        layers = model(batch)

    embedding_vectors = concatenate_layers(layers, device)

    if indices is not None:
        embedding_vectors = torch.index_select(embedding_vectors, 1, indices)

    return embedding_vectors



def extract_embedding_vectors_dataloader(model, dataloader, device, indices=None):

    embedding_vectors = None

    for (batch, _, _) in tqdm(dataloader, 'Feature extraction'):

        batch_embedding_vectors = extract_embedding_vectors(model, batch, device, indices=indices)

        if embedding_vectors is None:
            embedding_vectors = batch_embedding_vectors
        else:
            embedding_vectors = torch.cat((embedding_vectors, batch_embedding_vectors), 0)

    return embedding_vectors



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
