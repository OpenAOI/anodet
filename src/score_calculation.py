from scipy.spatial.distance import mahalanobis
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm




def calculateImageScore(patch_scores):
    image_scores = patch_scores.reshape(patch_scores.shape[0], -1).max(axis=1).values
    return image_scores


def calculateImageClassification(image_scores, thresh):
    image_classifications = image_scores.clone()
    image_classifications[image_classifications < thresh] = 1
    image_classifications[image_classifications >= thresh] = 0  
    return image_classifications


def calculatePatchClassification(patch_scores, thresh):
    patch_classifications = patch_scores.clone()
    patch_classifications[patch_classifications < thresh] = 1
    patch_classifications[patch_classifications >= thresh] = 0
    return patch_classifications
    


def calculatePatchScore(mean, cov_inv, embedding_vectors, device, do_gaussian_filter=True):

    # Reshape and switch axes to conform to mahalonobis function
    mean = mean.permute(1,0)
    cov_inv = cov_inv.permute(2, 0, 1)
    d,l,w,h = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape(d,l,w*h)
    embedding_vectors = embedding_vectors.permute(0,2,1)
    
    # Calculate distances
    mahalanobis_distances = mahalanobis(mean, cov_inv, embedding_vectors)
    
    # Reshape output
    patch_scores = mahalanobis_distances.reshape(d, w, h)
    if d == 1:
        patch_scores = patch_scores.squeeze(0)
        
    #TODO: pytorch implementation of gaussian filter
    # Apply gaussian filter
    if do_gaussian_filter:
        patch_scores = patch_scores.cpu().numpy()
        for i in range(patch_scores.shape[0]):
            patch_scores[i] = gaussian_filter(patch_scores[i], sigma=4)
        patch_scores = torch.from_numpy(patch_scores).to(device)

    return patch_scores
    
    
    
    
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
    mahalanobis_distance = sqrt.reshape(N)
    
    # If a set of sets of distances is to be computed, reshape output
    if N > n:
        mahalanobis_distance = mahalanobis_distance.reshape(ratio, n)
    
    return mahalanobis_distance



