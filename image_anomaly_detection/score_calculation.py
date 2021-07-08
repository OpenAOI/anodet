from scipy.spatial.distance import mahalanobis
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm



def calculateImageScore(patch_scores: torch.tensor):
    """Calculate image score
    
    Args:
        patch_scores: A batch of patch scores
        
    Returns:
        image_scores: A batch of image scores
    
    """
    
    # Calculate max value of each matrix
    image_scores = patch_scores.reshape(patch_scores.shape[0], -1).max(axis=1).values
    return image_scores


def calculateImageClassification(image_scores: torch.tensor, thresh: float) -> torch.tensor:
    """Calculate image classification
    
    Args:
        image_scores: A batch of iamge scores
        thresh: A treshold value. If an image score is larger than or equal to thresh it is classified as anomalous
        
    Returns:
        image_classifications: A batch of image classifcations
        
    """
    
    # Apply threshold
    image_classifications = image_scores.clone()
    image_classifications[image_classifications < thresh] = 1
    image_classifications[image_classifications >= thresh] = 0  
    return image_classifications


def calculatePatchClassification(patch_scores: torch.tensor, thresh: float) -> torch.tensor:
    """Calculate patch classification
    
    Args:
        patch_scores: A batch of patch scores
        thresh: A treshold value. If a patch score is larger than or equal to thresh it is classified as anomalous
        
    Returns:
        patch_classifications: A batch of patch classifications
    
    """
    
    # Apply threshold
    patch_classifications = patch_scores.clone()
    patch_classifications[patch_classifications < thresh] = 1
    patch_classifications[patch_classifications >= thresh] = 0
    return patch_classifications


def calculatePatchScore(mean: torch.tensor, cov_inv: torch.tensor, embedding_vectors: torch.tensor, device: torch.device, do_gaussian_filter: bool=True) -> torch.tensor:
    """Calculate the patch score for embedding vectors
    
    Args:
        mean: A batch of mean vectors
        cov_inv: A batch of inverted covariance matrices
        embedding_vectors: A batch of embedding vectors
        device: The device on which to run the function
        do_gaussian_filter: If True apply gaussian filter, else do not
        
    Returns:
        patch_scores: A batch of patch scores
    
    """
    
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
        
    #TODO: pytorch implementation of gaussian filter
    # Apply gaussian filter
    if do_gaussian_filter:
        patch_scores = patch_scores.cpu().numpy()
        for i in range(patch_scores.shape[0]):
            patch_scores[i] = gaussian_filter(patch_scores[i], sigma=4)
        patch_scores = torch.from_numpy(patch_scores).to(device)

    return patch_scores
    
    
def mahalanobis(mean: torch.tensor, cov_inv: torch.tensor, x: torch.tensor) -> torch.tensor:  
    """Calculate the mahalonobis distance
    
    Calculate the mahalanobis distance between a multivariate normal distribution 
    and a point or elementwise between a set of distributions and a set of points.
    
    Args:
        mean: A mean vector or a set of mean vectors
        cov_inv: A inverse of covariance matrix or a set of covariance matricies
        x: A point or a set of points
        
    Returns:
        mahalonobis_distance: A distance or a set of distances or a set of sets of distances
        
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

