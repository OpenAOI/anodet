import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import datasets.mvtec as mvtec


from PIL import Image
from torchvision import transforms as T

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


import cv2



class AnomalyDetector:
    
    def __init__(self, path, device=None):
        
        if device == None:
            self.use_cuda = torch.cuda.is_available()
            self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        else:
            self.device = device
            
            
        self.model = wide_resnet50_2(pretrained=True, progress=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.mean, self.cov = loadFeatures(path)
        self.train_outputs = [self.mean, self.cov]

    
    
    def __call__(self, images,thresh=6):
        batch = toBatch(images)
        batch_embedding_vectors = extractEmbeddingVectorsBatched(batch, self.model, self.device)
        score_maps = calculateScore(self.train_outputs, batch_embedding_vectors)

        img_scores = score_maps.reshape(score_maps.shape[0], -1).max(axis=1)
        print('--------------')
        print(img_scores)
        img_scores[np.where(img_scores < thresh)] = True
        img_scores[np.where(img_scores >= thresh)] = False
#         img_scores = img_scores.astype(bool)
        img_scores = list(img_scores)
        print(img_scores)
        res = []
        for s in img_scores:
            res.append(int(s))
        

        image = getBatchVisImage(images, score_maps, thresh, res)

        
        return res, image
        
        
        



def getBatchVisImage(images, score_maps, thresh, scores):

    indent = 20
    padding = 30
    
    max_height = 0
    tot_width = 0
    for image in images:
        if image.shape[0] > max_height:
            max_height = image.shape[0]
        tot_width += image.shape[1]
    
    

    
    image_tot = np.ones((max_height+3*indent, tot_width + padding*(len(images))+ 2*indent*(len(images)+1), 3))*255
    images = images[::-1]
    score_maps = score_maps[::-1]
    scores = scores[::-1]
    
    last_x = 0
    for i in range(len(images)):

        frame = np.zeros((images[i].shape[1]+2*indent, images[i].shape[0]+2*indent, 3))
        if scores[i]==0:
            frame[:,:,0] = np.ones((images[i].shape[1]+2*indent, images[i].shape[0]+2*indent))*255
        else:
            frame[:,:,1] = np.ones((images[i].shape[1]+2*indent, images[i].shape[0]+2*indent))*255
        

        mask = score_maps[i].copy()
        mask[mask > thresh] = 255
        mask[mask <= thresh] = 0

#         resized = cv2.resize(images[i], (56,56), interpolation = cv2.INTER_AREA)
        image = images[i].copy()


        transparent = np.zeros(mask.shape)
        line_img = mark_boundaries(transparent, mask, color=(1, 0, 0), mode='thick')
        line_img = cv2.resize(line_img, (image.shape[1],image.shape[0]), interpolation = cv2.INTER_AREA)
        line_img = (line_img*255).astype(np.uint8)
        

        a = line_img == [255,0,0]
        a = a[:,:,0]
    

        image[:,:,0] = np.where(a, line_img[:,:,0], image[:,:,0])
        image[:,:,1] = np.where(a, 0, image[:,:,1])
        image[:,:,2] = np.where(a, 0, image[:,:,1])
        
        
#         vis_img = (vis_img*255).astype(np.uint8)


        frame[indent:indent+image.shape[0],indent:indent+image.shape[1]] = image
        
        image_tot[0:frame.shape[0], last_x:last_x + frame.shape[1],:] = frame
        last_x = last_x + frame.shape[1]+padding
        image_tot = image_tot.astype(np.uint8)
        

    return image_tot





def extractEmbeddingVectors(dataloader, model, device):

    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)    

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    for (x, _, _) in tqdm(dataloader, '| feature extraction | train |'):
        # model prediction
        
        with torch.no_grad():
            _ = model(x.to(device))
        # get intermediate layer outputs
        for k, v in zip(train_outputs.keys(), outputs):
            train_outputs[k].append(v.cpu().detach())

        # initialize hook outputs
        outputs = []
        
    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)


    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
        if 'layer1' in train_outputs:
            del train_outputs['layer1']
        del train_outputs[layer_name]
        
    idx = getSeedIndices(100, 448, device)  
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)

    return embedding_vectors



def extractEmbeddingVectorsBatched(x, model, device):

    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)    
#     model.layer4[-1].register_forward_hook(hook)    

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])


    with torch.no_grad():
        _ = model(x.to(device))
    # get intermediate layer outputs
    for k, v in zip(train_outputs.keys(), outputs):
        train_outputs[k].append(v.cpu().detach())

    # initialize hook outputs
    outputs = []
        
    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)


    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
        
    idx = getSeedIndices(100, 448, device)  
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    
    return embedding_vectors




def getSeedIndices(choose, total, device):
    random.seed(1024)
    torch.manual_seed(1024)
        
    if device.type=='cuda':
        torch.cuda.manual_seed_all(1024)
    return torch.tensor(sample(range(0, total), choose))

def getRandomIndices(choose, total):
    pass

def getLowVarIndices(choose, total, features):
    pass
    



def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    del x
    del y
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z



def getParameters(embedding_vectors):
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)

    # embedding_vectors = torch.from_numpy(embedding_vectors)

    for i in range(H * W):
        # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        
    return mean, cov




def saveFeatures(features, path):
    with open(path, 'wb') as f:
        pickle.dump(features, f)
        
        
def loadFeatures(path):
    with open(path, 'rb') as f:
        features = pickle.load(f)
    return features




def toBatch(images):
    resize = 256
    cropsize = 224
    transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                  T.CenterCrop(cropsize),
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                            ])

    batch = None
    
    for image in images:
        x = Image.fromarray(image).convert('RGB')
        x = transform_x(x).unsqueeze(0)
        
        if batch == None:
            batch = x
        else:
            batch = torch.cat((batch, x), 0)
                
    return batch


def loadBatch(PATHS):
    resize = 256
    cropsize = 224
    transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                  T.CenterCrop(cropsize),
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                            ])

    batch = None
    
    for path in PATHS:
        x = Image.open(path).convert('RGB')
        x = transform_x(x).unsqueeze(0)
        
        if batch == None:
            batch = x
        else:
            batch = torch.cat((batch, x), 0)
                
    return batch
        
    
def calculateScore(train_outputs, embedding_vectors):

    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    dist_list = torch.tensor(dist_list)
#     print(embedding_vectors.shape)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=H, mode='bilinear',
                              align_corners=False).squeeze().numpy()#TODO: Check size

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
#     max_score = score_map.max()
#     min_score = score_map.min()
#     scores = (score_map - min_score) / (max_score - min_score)
    return score_map