"""
Provides utility functions for anomaly detection.
"""

import numpy as np
import torch
from typing import List
from torchvision import transforms as T
from PIL import Image
import os


standard_image_transform = T.Compose([T.Resize(224),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
                                      ])

standard_mask_transform = T.Compose([T.Resize(224),
                                     T.CenterCrop(224),
                                     T.ToTensor()
                                     ])


def to_batch(images: List[np.ndarray], transforms: T.Compose, device: torch.device) -> torch.Tensor:
    """Convert a list of numpy array images to a pytorch tensor batch with given transforms."""
    assert len(images) > 0

    transformed_images = []
    for i, image in enumerate(images):
        image = Image.fromarray(image).convert('RGB')
        transformed_images.append(transforms(image))

    height, width = transformed_images[0].shape[1:3]
    batch = torch.zeros((len(images), 3, height, width))

    for i, transformed_image in enumerate(transformed_images):
        batch[i] = transformed_image

    return batch.to(device)


# From: https://github.com/pytorch/pytorch/issues/19037
def pytorch_cov(tensor: torch.Tensor, rowvar: bool = True, bias: bool = False) -> torch.Tensor:
    """Estimate a covariance matrix (np.cov)."""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def mahalanobis(mean: torch.Tensor, cov_inv: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Calculate the mahalonobis distance

    Calculate the mahalanobis distance between a multivariate normal distribution
    and a point or elementwise between a set of distributions and a set of points.

    Args:
        mean: A mean vector or a set of mean vectors.
        cov_inv: A inverse of covariance matrix or a set of covariance matricies.
        batch: A point or a set of points.

    Returns:
        mahalonobis_distance: A distance or a set of distances or a set of sets of distances.

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


def image_score(patch_scores: torch.Tensor) -> torch.Tensor:
    """Calculate image scores from patch scores.

    Args:
        patch_scores: A batch of patch scores.

    Returns:
        image_scores: A batch of image scores.

    """

    # Calculate max value of each matrix
    image_scores = torch.max(patch_scores.reshape(patch_scores.shape[0], -1), -1).values
    return image_scores


def classification(image_scores: torch.Tensor, thresh: float) -> torch.Tensor:
    """Calculate image classifications from image scores.

    Args:
        image_scores: A batch of image scores.
        thresh: A treshold value. If an image score is larger than
                or equal to thresh it is classified as anomalous.

    Returns:
        image_classifications: A batch of image classifcations.

    """

    # Apply threshold
    image_classifications = image_scores.clone()
    image_classifications[image_classifications < thresh] = 1
    image_classifications[image_classifications >= thresh] = 0
    return image_classifications


def get_paths_for_directory_path(directory_path: str, limit: int = None) -> List:
    """Get paths for files in a folder.
    Args:
        directory_path: Path to folder.
        limit: Specific amount of paths, leave empty for all
    Returns:
        file_paths: List of paths as a string.
    """
    file_paths = []

    for i, file in enumerate(os.listdir(directory_path)):
        if limit is not None and i >= limit:
            break
        filename = os.fsdecode(file)
        file_paths.append(os.path.join(directory_path, filename))

    return file_paths
