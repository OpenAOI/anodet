"""
Provides functions for calculating the multivariate normal distribution of embedding vectors
"""

import typing
import torch
import numpy as np
from tqdm import tqdm



def joint_normal_distribution(embedding_vectors: torch.Tensor, device: torch.device,
                              invert_cov: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Calculate multivariate normal distribution from embedding vectors

    Args:
        embedding_vectors: A batch of embedding vectors
        device: The device on which to run the function

    Returns:
        (mean, cov_inv): The mean vectors and the inverted covariance matrices

    """

    batch_length, channels, height, width = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(batch_length, channels, height * width)
    print('Calculating mean')
    mean = torch.mean(embedding_vectors, dim=0)#.numpy()
    cov = torch.zeros(channels, channels, height * width).numpy()
    identity_matrix = np.identity(channels)

    for i in tqdm(range(height*width), 'Calculating covariance'):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False) \
        + 0.01 * identity_matrix

    cov = torch.from_numpy(cov).to(device)
    if not invert_cov:
        return mean, cov

    print('Calculating inverse of covariance')
    cov_inv = cov.permute(2, 0, 1)
    cov_inv = torch.inverse(cov_inv)
    cov_inv = cov_inv.permute(1, 2, 0)

    return mean, cov_inv
