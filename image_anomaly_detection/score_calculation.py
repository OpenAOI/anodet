"""
Provides functions for calculating different scores based on distributions and embedding vectors
"""

import torch
import torchvision



def image_score(patch_scores: torch.Tensor) -> torch.Tensor:
    """Calculate image scores from patch scores

    Args:
        patch_scores: A batch of patch scores

    Returns:
        image_scores: A batch of image scores

    """

    # Calculate max value of each matrix
    image_scores = torch.max(patch_scores.reshape(patch_scores.shape[0], -1), -1).values
    return image_scores



def image_classification(image_scores: torch.Tensor, thresh: float) -> torch.Tensor:
    """Calculate image classifications from image scores

    Args:
        image_scores: A batch of iamge scores
        thresh: A treshold value. If an image score is larger than
                or equal to thresh it is classified as anomalous

    Returns:
        image_classifications: A batch of image classifcations

    """

    # Apply threshold
    image_classifications = image_scores.clone()
    image_classifications[image_classifications < thresh] = 1
    image_classifications[image_classifications >= thresh] = 0
    return image_classifications



def patch_classification(patch_scores: torch.Tensor, thresh: float) -> torch.Tensor:
    """Calculate patch classifications from patch scores

    Args:
        patch_scores: A batch of patch scores
        thresh: A treshold value. If a patch score is larger than
                or equal to thresh it is classified as anomalous

    Returns:
        patch_classifications: A batch of patch classifications

    """

    # Apply threshold
    patch_classifications = patch_scores.clone()
    patch_classifications[patch_classifications < thresh] = 1
    patch_classifications[patch_classifications >= thresh] = 0
    return patch_classifications



def patch_score(mean: torch.Tensor, cov_inv: torch.Tensor,
                embedding_vectors: torch.Tensor,
                apply_gaussian_filter: bool = True) -> torch.Tensor:
    """Calculate patch scores from embedding vectors

    Args:
        mean: A batch of mean vectors
        cov_inv: A batch of inverted covariance matrices
        embedding_vectors: A batch of embedding vectors
        do_gaussian_filter: If True apply gaussian filter, else do not

    Returns:
        patch_scores: A batch of patch scores

    """

    # Reshape and switch axes to conform to mahalonobis function
    mean = mean.permute(1, 0)
    cov_inv = cov_inv.permute(2, 0, 1)
    batch_size, length, width, height = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape(batch_size, length, width*height)
    embedding_vectors = embedding_vectors.permute(0, 2, 1)

    # Calculate distances
    mahalanobis_distances = mahalanobis(mean, cov_inv, embedding_vectors)

    # Reshape output
    patch_scores = mahalanobis_distances.reshape(batch_size, width, height)

    # Apply gaussian filter
    if apply_gaussian_filter:
        patch_scores = torchvision.transforms.GaussianBlur(33, sigma=4)(patch_scores)

    return patch_scores



def mahalanobis(mean: torch.Tensor, cov_inv: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Calculate the mahalonobis distance

    Calculate the mahalanobis distance between a multivariate normal distribution
    and a point or elementwise between a set of distributions and a set of points.

    Args:
        mean: A mean vector or a set of mean vectors
        cov_inv: A inverse of covariance matrix or a set of covariance matricies
        batch: A point or a set of points

    Returns:
        mahalonobis_distance: A distance or a set of distances or a set of sets of distances

    """

    # Assert that parameters has acceptable dimensions
    assert len(mean.shape) == 1 or len(mean.shape) == 2, \
    'mean must be a vector or a set of vectors (matrix)'
    assert len(batch.shape) == 1 or len(batch.shape) == 2 or len(batch.shape) == 3, \
    'batch must be a vector or a set of vectors (matrix) or a set of sets of vectors (3d tensor)'
    assert len(cov_inv.shape) == 2 or len(cov_inv.shape) == 3, \
    'cov_inv must be a matrix or a set of matrices (3d tensor)'

    # Standardize the dimensions
    if len(mean.shape) == 1:
        mean = mean.unsqueeze(0)
    if len(cov_inv.shape) == 2:
        cov_inv = cov_inv.unsqueeze(0)
    if len(batch.shape) == 1:
        batch = batch.unsqueeze(0)
    if len(batch.shape) == 3:
        batch = batch.reshape(batch.shape[0]*batch.shape[1], batch.shape[2])

    # Assert that parameters has acceptable shapes
    assert mean.shape[0] == cov_inv.shape[0]
    assert mean.shape[1] == cov_inv.shape[1] == cov_inv.shape[2] == batch.shape[1]
    assert batch.shape[0] % mean.shape[0] == 0

    # Set shape variables
    mini_batch_size, length = mean.shape
    batch_size = batch.shape[0]
    ratio = int(batch_size/mini_batch_size)

    # If a set of sets of distances is to be computed, expand mean and cov_inv
    if batch_size > mini_batch_size:
        mean = mean.unsqueeze(0)
        mean = mean.expand(ratio, mini_batch_size, length)
        mean = mean.reshape(batch_size, length)
        cov_inv = cov_inv.unsqueeze(0)
        cov_inv = cov_inv.expand(ratio, mini_batch_size, length, length)
        cov_inv = cov_inv.reshape(batch_size, length, length)

    # Make sure tensors are correct type
    mean = mean.float()
    cov_inv = cov_inv.float()
    batch = batch.float()

    # Calculate mahalanobis distance
    diff = mean-batch
    mult1 = torch.bmm(diff.unsqueeze(1), cov_inv)
    mult2 = torch.bmm(mult1, diff.unsqueeze(2))
    sqrt = torch.sqrt(mult2)
    mahalanobis_distance = sqrt.reshape(batch_size)

    # If a set of sets of distances is to be computed, reshape output
    if batch_size > mini_batch_size:
        mahalanobis_distance = mahalanobis_distance.reshape(ratio, mini_batch_size)

    return mahalanobis_distance
