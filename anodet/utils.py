"""
Provides utility functions for anomaly detection.
"""


import os
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Used in AnodetDataset class
standard_image_transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[1.485, 1.456, 1.406], std=[1.229, 1.224, 1.225]),
        #TODO Why are we normalizing with these values?
    ]
)

standard_mask_transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])


def to_batch(
    images: List[np.ndarray], transforms: T.Compose, device: torch.device
) -> torch.Tensor:
    """Convert a list of numpy array images to a pytorch tensor batch with given transforms."""
    assert len(images) > 0

    transformed_images = []
    for i, image in enumerate(images):
        image = Image.fromarray(image).convert("RGB")
        transformed_images.append(transforms(image))

    height, width = transformed_images[0].shape[1:3]
    batch = torch.zeros((len(images), 3, height, width))

    for i, transformed_image in enumerate(transformed_images):
        batch[i] = transformed_image

    return batch.to(device)


# From: https://github.com/pytorch/pytorch/issues/19037
def pytorch_cov(
    tensor: torch.Tensor, rowvar: bool = True, bias: bool = False
) -> torch.Tensor:
    """Estimate a covariance matrix (np.cov)."""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def mahalanobis(
    mean: torch.Tensor, cov_inv: torch.Tensor, batch: torch.Tensor
) -> torch.Tensor:
    """Calculate the mahalanobis distance

    Calculate the mahalanobis distance between a multivariate normal distribution
    and a point or elementwise between a set of distributions and a set of points.

    Args:
        mean: A mean vector or a set of mean vectors.
        cov_inv: A inverse of covariance matrix or a set of covariance matricies.
        batch: A point or a set of points.

    Returns:
        mahalanobis_distance: A distance or a set of distances or a set of sets of distances.

    """

    # Assert that parameters has acceptable dimensions
    assert (
        len(mean.shape) == 1 or len(mean.shape) == 2
    ), "mean must be a vector or a set of vectors (matrix)"
    assert (
        len(batch.shape) == 1 or len(batch.shape) == 2 or len(batch.shape) == 3
    ), "batch must be a vector or a set of vectors (matrix) or a set of sets of vectors (3d tensor)"
    assert (
        len(cov_inv.shape) == 2 or len(cov_inv.shape) == 3
    ), "cov_inv must be a matrix or a set of matrices (3d tensor)"

    # Standardize the dimensions
    if len(mean.shape) == 1:
        mean = mean.unsqueeze(0)
    if len(cov_inv.shape) == 2:
        cov_inv = cov_inv.unsqueeze(0)
    if len(batch.shape) == 1:
        batch = batch.unsqueeze(0)
    if len(batch.shape) == 3:
        batch = batch.reshape(batch.shape[0] * batch.shape[1], batch.shape[2])

    # Assert that parameters has acceptable shapes
    assert mean.shape[0] == cov_inv.shape[0]
    assert mean.shape[1] == cov_inv.shape[1] == cov_inv.shape[2] == batch.shape[1]
    assert batch.shape[0] % mean.shape[0] == 0

    # Set shape variables
    mini_batch_size, length = mean.shape
    batch_size = batch.shape[0]
    ratio = int(batch_size / mini_batch_size)

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
    diff = mean - batch
    mult1 = torch.bmm(diff.unsqueeze(1), cov_inv)
    mult2 = torch.bmm(mult1, diff.unsqueeze(2))
    sqrt = torch.sqrt(mult2)
    mahalanobis_distance = sqrt.reshape(batch_size)

    # Visualize the 'diff' tensor
    # This can help in understanding the differences between the mean and batch vectors, providing insights into the direction and magnitude of deviation.
    # TODO Make function for this viz
    plt.figure()
    plt.plot(diff.detach().numpy())
    plt.title('Differences between Mean and Batch Vectors')
    plt.xlabel('Vector Components')
    plt.ylabel('Difference Magnitude')
    plt.show()


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


def rename_files(source_path: str, destination_path: Optional[str] = None) -> None:
    """Rename all files in a directory path with increasing integer name.
    Ex. 0001.png, 0002.png ...
    Write files to destination path if argument is given.

    Args:
        source_path: Path to folder.
        destination_path: Path to folder.

    """
    for count, filename in enumerate(os.listdir(source_path), 1):
        file_source_path = os.path.join(source_path, filename)
        file_extension = os.path.splitext(filename)[1]

        new_name = str(count).zfill(4) + file_extension
        if destination_path:
            new_destination = os.path.join(destination_path, new_name)
        else:
            new_destination = os.path.join(source_path, new_name)

        os.rename(file_source_path, new_destination)


def split_tensor_and_run_function(
    func: Callable[[torch.Tensor], List],
    tensor: torch.Tensor,
    split_size: Union[int, List],
) -> torch.Tensor:
    """Splits the tensor into chunks in given split_size and run a function on each chunk.

    Args:
        func: Function to be run on a chunk of tensor.
        tensor: Tensor to split.
        split_size: Size of a single chunk or list of sizes for each chunk.

    Returns:
        output_tensor: Tensor of same size as input tensor

    """
    tensors_list = []
    for sub_tensor in torch.split(tensor, split_size):
        tensors_list.append(func(sub_tensor))

    output_tensor = torch.cat(tensors_list)

    return output_tensor


if __name__ == "__main__":

    mean = torch.tensor([1,2,3])
    cov_inv = torch.tensor([[1,0,0], [0,1,0], [0,0,1]])
    batch = torch.tensor([[2,3,4], [1,6,7]])
    expected_results = torch.tensor([1.722, 5.1962])
    
    result = mahalanobis(mean, cov_inv, batch)
    print(result)

    # Generate synthetic data for testing
    num_samples = 100
    num_features = 10

    mean = torch.rand(num_features)
    cov = torch.rand(num_features, num_features)
    cov_inv = torch.inverse(cov)

    batch = torch.rand(num_samples, num_features)

    # Test the mahalanobis function with the generated data
    result = mahalanobis(mean, cov_inv, batch)

    # Print the result or perform additional tests
    print(result)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sample_indices = range(len(result))
    ax.scatter(sample_indices, result[:, 0], np.zeros_like(sample_indices), c='b', marker='o')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Mahalanobis Distance')
    ax.set_zlabel('Zero')
    ax.set_title('Mahalanobis Distances for Samples')
    plt.show()


    ## ----- # 

    #  Two clusters
    
    # Parameters for the two clusters
    num_samples_cluster1 = 100
    num_samples_cluster2 = 100
    num_features = 3

    # Generate data points for the two clusters
    mean_cluster1 = torch.tensor([2, 2, 2])
    cov_cluster1 = torch.tensor([[1, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1]])
    cluster1_data = np.random.multivariate_normal(mean_cluster1, cov_cluster1, num_samples_cluster1)

    mean_cluster2 = torch.tensor([-2, -2, -2])
    cov_cluster2 = torch.tensor([[1, -0.5, 0.3], [-0.5, 1, 0.2], [0.3, 0.2, 1]])
    cluster2_data = np.random.multivariate_normal(mean_cluster2, cov_cluster2, num_samples_cluster2)

    # Concatenate the data points from the two clusters
    batch = np.concatenate((cluster1_data, cluster2_data), axis=0)

    # Calculate Mahalanobis distance for the combined data
    mean = torch.tensor([0, 0, 0])  # Choose a mean for the function
    cov_inv = torch.inverse(torch.eye(num_features))  # Choose an inverse covariance for the function

    distances = mahalanobis(mean, cov_inv, torch.tensor(batch))

    # Plot the clusters and Mahalanobis distances on a 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cluster1_data[:, 0], cluster1_data[:, 1], cluster1_data[:, 2], c='b', marker='o', label='Cluster 1')
    ax.scatter(cluster2_data[:, 0], cluster2_data[:, 1], cluster2_data[:, 2], c='r', marker='^', label='Cluster 2')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Two Separate Clusters on 3D Axis')

    # Visualize Mahalanobis distances as text annotations
    distances_np = distances.detach().numpy()  # Convert to a NumPy array
    for i in range(len(batch)):
        ax.text(batch[i][0], batch[i][1], batch[i][2], f'{distances_np[i][0]:.2f}', color='black')

    ax.legend()
    plt.show()