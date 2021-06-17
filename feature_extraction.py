import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
from tqdm import tqdm




class Resnet18Features(nn.Module):

    def __init__(self, device):
        super(Resnet18Features, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
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
    
    
    
    
def concatenateLayers(layers):
    
    tot_depth = sum(layer.shape[1] for layer in layers)
    
    #TODO: Check if this takes up more memory, probably better for speed
    concatenated_layers = torch.zeros((layers[0].shape[0], tot_depth, layers[0].shape[2], layers[0].shape[3]))
    concatenated_layers[:, 0:layers[0].shape[1], :, :] = layers[0]

    last_depth = layers[0].shape[0]
    
    for layer in layers[1:]:
        scale_factor = concatenated_layers.shape[3]/layer.shape[3]
        upsampled_layer = torch.nn.Upsample(scale_factor=scale_factor, mode='nearest')(layer)
        concatenated_layers[:,last_depth:last_depth+upsampled_layer.shape[1], :, :] = upsampled_layer
        last_depth += upsampled_layer.shape[1]
    
    return concatenated_layers
    
    


def extractEmbeddingVectors(model, x):
    with torch.no_grad():
        layers = model(x)
    
    embedding_vectors = concatenateLayers(layers)
#     B,D,H,W = embedding_vectors.shape
#     embedding_vectors = embedding_vectors.reshape(B, D, H*W)
    return embedding_vectors
    
    
    
    
def extractEmbeddingVectorsDataloader(model, dataloader):
    
    embedding_vectors = None
        
    for (x, _, _) in tqdm(dataloader, 'Feature extraction'):
        with torch.no_grad():
            layers = model(x)
        
        batch_embedding_vectors = concatenateLayers(layers)
        if embedding_vectors == None:
            embedding_vectors = batch_embedding_vectors
        else:
            embedding_vectors = torch.cat((embedding_vectors, batch_embedding_vectors), 0)
            
#     B,D,H,W = embedding_vectors.shape
#     embedding_vectors = embedding_vectors.reshape(B, D, H*W)
    return embedding_vectors