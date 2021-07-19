"""
Provides utility functions for anomaly detection
"""

from typing import List
import torch
from torchvision import transforms as T
from PIL import Image
import numpy as np


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
    """Converts a list of numpy array images to a pytorch tensor batch with given transforms"""
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
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


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
