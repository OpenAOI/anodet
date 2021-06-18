import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm
import random
from random import sample




class WideResnet50Features(nn.Module):

    def __init__(self, device):
        super(WideResnet50Features, self).__init__()
        self.resnet50_2 = models.wide_resnet50_2(pretrained=True, progress=True)
        self.resnet50_2.to(device)
        self.eval()
        
    def eval(self):
        self.resnet50_2.eval()
     
    def to(self, device):
        self.resnet50_2.to(device)
    
    def forward(self, x):
        x = self.resnet50_2.conv1(x)
        x = self.resnet50_2.bn1(x)
        x = self.resnet50_2.relu(x)
        prelayer_out = self.resnet50_2.maxpool(x)
        
        layer1 = self.resnet50_2.layer1(prelayer_out)
        layer2 = self.resnet50_2.layer2(layer1)
        layer3 = self.resnet50_2.layer3(layer2)
        #layer4_out = self.resnet18.layer4(layer3)
                
        return layer1, layer2, layer3


class Resnet18Features(nn.Module):

    def __init__(self, device):
        super(Resnet18Features, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True, progress=True)
        self.resnet18.to(device)
        self.eval()
        
    def eval(self):
        self.resnet18.eval()
     
    def to(self, device):
        self.resnet18.to(device)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        prelayer_out = self.resnet18.maxpool(x)
        
        layer1 = self.resnet18.layer1(prelayer_out)
        layer2 = self.resnet18.layer2(layer1)
        layer3 = self.resnet18.layer3(layer2)
        #layer4_out = self.resnet18.layer4(layer3)
                
        return layer1, layer2, layer3
    
    
    
    
def concatenateLayers(features, device):
    expanded_features = features[0]
    for feature in features[1:]:
        expanded_features = embedding_concat(expanded_features, feature, device)
    return expanded_features
    


def embedding_concat(x, y, device):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2, device=device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    del x
    del y
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z
    
    
    
# def concatenateLayers(layers, device):
    
#     tot_depth = sum(layer.shape[1] for layer in layers)
    
#     #TODO: Check if this takes up more memory, probably better for speed
#     concatenated_layers = torch.zeros((layers[0].shape[0], tot_depth, layers[0].shape[2], layers[0].shape[3]), device=device)
#     concatenated_layers[:, 0:layers[0].shape[1], :, :] = layers[0]

#     last_depth = layers[0].shape[0]
    
#     for layer in layers[1:]:
#         scale_factor = concatenated_layers.shape[3]/layer.shape[3]
#         upsampled_layer = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')(layer)
#         concatenated_layers[:,last_depth:last_depth+upsampled_layer.shape[1], :, :] = upsampled_layer
#         last_depth += upsampled_layer.shape[1]
    
#     return concatenated_layers
    
    


def extractEmbeddingVectors(model, x, device):
    with torch.no_grad():
        x = x.to(device)
        layers = model(x)
    
    embedding_vectors = concatenateLayers(layers, device)
#     B,D,H,W = embedding_vectors.shape
#     embedding_vectors = embedding_vectors.reshape(B, D, H*W)
    return embedding_vectors
    
    
    
    
def extractEmbeddingVectorsDataloader(model, dataloader, device):
    
    embedding_vectors = None
        
    for (x, _, _) in tqdm(dataloader, 'Feature extraction'):
        x = x.to(device)
        with torch.no_grad():
            layers = model(x)
        
        batch_embedding_vectors = concatenateLayers(layers, device)
        if embedding_vectors == None:
            embedding_vectors = batch_embedding_vectors
        else:
            embedding_vectors = torch.cat((embedding_vectors, batch_embedding_vectors), 0)
            
#     B,D,H,W = embedding_vectors.shape
#     embedding_vectors = embedding_vectors.reshape(B, D, H*W)

    return embedding_vectors






def getIndices(choose, total, device):
    random.seed(1024)
    torch.manual_seed(1024)
        
    if device.type=='cuda':
        torch.cuda.manual_seed_all(1024)
        
    return torch.tensor(sample(range(0, total), choose), device=device)