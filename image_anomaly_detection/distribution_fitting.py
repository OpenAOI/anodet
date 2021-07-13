"""
Provides functions for calculating the multivariate normal distribution of embedding vectors
"""

import typing
import torch



def joint_normal_distribution(embedding_vectors:
                              torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Calculate multivariate normal distribution from embedding vectors

    Args:
        embedding_vectors: A batch of embedding vectors

    Returns:
        (mean, cov_inv): The mean vectors and the inverted covariance matrices

    """

    # Prepare embedding vectors
    batch_length, channels, height, width = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(batch_length, channels, height * width)

    # Calculate mean
    mean = torch.mean(embedding_vectors, dim=0)

    # Calculate covariance
    identity = torch.eye(channels)
    cov = pytorch_cov(embedding_vectors.permute(2, 0, 1), rowvar=False) + 0.01 * identity

    # Calculate inverse
    cov_inv = torch.inverse(cov)
    cov_inv = cov_inv.permute(1, 2, 0)

    return mean, cov_inv



# From: https://github.com/pytorch/pytorch/issues/19037
def pytorch_cov(tensor: torch.Tensor, rowvar: bool=True, bias: bool=False) -> torch.Tensor:
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()
