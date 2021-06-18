from scipy.spatial.distance import mahalanobis
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm




def calculateScoreMaps(mean, cov_inv, embedding_vectors, device, do_gaussian_filter=True):

    # Reshape and switch axes to conform to mahalonobis function
    mean = mean.permute(1,0)
    cov_inv = cov_inv.permute(2, 0, 1)
    d,l,w,h = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape(d,l,w*h)
    embedding_vectors = embedding_vectors.permute(0,2,1)
    
    # Calculate distances
    mahalanobis_distances = mahalanobis(mean, cov_inv, embedding_vectors)
    
    # Reshape output
    score_maps = mahalanobis_distances.reshape(d, w, h)
    if d == 1:
        score_maps = score_maps.squeeze(0)
        
    #TODO: pytorch implementation of gaussian filter
    # Apply gaussian filter
    if do_gaussian_filter:
        score_maps = score_maps.cpu().numpy()
        for i in range(score_maps.shape[0]):
            score_maps[i] = gaussian_filter(score_maps[i], sigma=4)
        score_maps = torch.from_numpy(score_maps).to(device)

    return score_maps
    
    
    
    
    
def mahalanobis(mean, cov_inv, x):  
    """
    Calculates the mahalanobis distance between a multivariate normal distribution 
    and a point or elementwise between a set of distributions and a set of points.
    
    If a set of points is to be calculated the function expects the first dimension to be the number of points for mean, cov_inv and x.
    
    Parameters:
        - mean: Mean vector. A pytorch vector or a set of vectors (matrix)
        - cov_inv: Inverse of covariance matrix. A pytorch matrix or a set of matrices (3d tensor)
        - x: Point. A pytorch vector or a set of vectors (matrix)
        
    Returns:
        - mahalonobis_distance: a pytorch value or a set of values (vector) or a set of vectors (matrix)
    """
    
    # Assert that parameters has acceptable dimensions
    assert len(mean.shape) == 1 or len(mean.shape) == 2, 'mean must be a vector or a set of vectors (matrix)'
    assert len(x.shape) == 1 or len(x.shape) == 2 or len(x.shape) == 3, 'x must be a vector or a set of vectors (matrix) or a set of sets of vectors (3d tensor)'
    assert len(cov_inv.shape) == 2 or len(cov_inv.shape) == 3, 'cov_inv must be a matrix or a set of matrices (3d tensor)'
        
    # Standardize the dimensions
    if len(mean.shape) == 1: 
        mean = mean.unsqueeze(0)
    if len(cov_inv.shape) == 2: 
        cov_inv = cov_inv.unsqueeze(0)
    if len(x.shape) == 1: 
        x = x.unsqueeze(0)        
    if len(x.shape) == 3: 
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2])
    
    # Assert that parameters has acceptable shapes
    assert mean.shape[0] == cov_inv.shape[0]
    assert mean.shape[1] == cov_inv.shape[1] == cov_inv.shape[2] == x.shape[1]
    assert x.shape[0] % mean.shape[0] == 0
    
    # Set shape variables
    n, l = mean.shape
    N = x.shape[0]
    ratio = int(N/n)
    
    # If a set of sets of distances is to be computed, expand mean and cov_inv
    if N > n:
        mean = mean.unsqueeze(0)
        mean = mean.expand(ratio, n, l)
        mean = mean.reshape(N, l)
        cov_inv = cov_inv.unsqueeze(0)
        cov_inv = cov_inv.expand(ratio, n, l, l)
        cov_inv = cov_inv.reshape(N, l, l)
    
    # Make sure tensors are correct type
    mean = mean.float()
    cov_inv = cov_inv.float()
    x = x.float()
    
    # Calculate mahalanobis distance
    diff = mean-x
    mult1 = torch.bmm(diff.unsqueeze(1), cov_inv)
    mult2 = torch.bmm(mult1, diff.unsqueeze(2))
    sqrt = torch.sqrt(mult2)
    mahalonobis_distance = sqrt.reshape(N)
    
    # If a set of sets of distances is to be computed, reshape output
    if N > n:
        mahalonobis_distance = mahalonobis_distance.reshape(ratio, n)
    
    return mahalonobis_distance







# def calculateScoreMaps(mean_, cov_inv_, embedding_vectors, do_gaussian_filter=True):
#     mean_ = mean_.cpu().numpy()
#     cov_inv_ = cov_inv_.cpu().numpy()
    
#     B, C, H, W = embedding_vectors.size()
#     embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()
#     dist_list = []
#     for i in range(H * W):
#         mean = mean_[:, i]
#         conv_inv = cov_inv_[:, :, i]
#         dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
#         dist_list.append(dist)

#     dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

#     # upsample
#     dist_list = torch.tensor(dist_list)
# #     print(embedding_vectors.shape)
#     score_map = F.interpolate(dist_list.unsqueeze(1), size=H, mode='bilinear', align_corners=False).squeeze().numpy()#TODO: Check size

#     # apply gaussian smoothing on the score map
#     if do_gaussian_filter:
#         for i in range(score_map.shape[0]):
#             score_map[i] = gaussian_filter(score_map[i], sigma=4)

        
#     score_map = torch.from_numpy(score_map)
#     return score_map




    

    
#TODO: Does not work on one single image
def calculateImageScores(score_maps, thresh):
    image_max_values = score_maps.reshape(score_maps.shape[0], -1).max(axis=1)
    img_scores = image_max_values.copy()
    img_scores[np.where(img_scores < thresh)] = True
    img_scores[np.where(img_scores >= thresh)] = False
    img_scores = list(img_scores)
    
    for i in range(len(img_scores)):
        img_scores[i] = int(img_scores[i])
    
    return image_max_values, img_scores
    


# def calculateScore(train_outputs, embedding_vectors):

#     B, C, H, W = embedding_vectors.size()
#     embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
#     dist_list = []
#     for i in tqdm(range(H*W), 'Calculating distances'):
#         mean = train_outputs[0][:, i]
#         conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
#         dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
#         dist_list.append(dist)

#     dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

#     # upsample
#     dist_list = torch.tensor(dist_list)
# #     print(embedding_vectors.shape)
#     score_map = F.interpolate(dist_list.unsqueeze(1), size=H, mode='bilinear',
#                               align_corners=False).squeeze().numpy()#TODO: Check size

#     # apply gaussian smoothing on the score map
#     for i in range(score_map.shape[0]):
#         score_map[i] = gaussian_filter(score_map[i], sigma=4)

#     # Normalization
# #     max_score = score_map.max()
# #     min_score = score_map.min()
# #     scores = (score_map - min_score) / (max_score - min_score)
#     return score_map