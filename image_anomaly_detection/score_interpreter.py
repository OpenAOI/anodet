"""
Provides functions for calculating different scores based on distributions and embedding vectors
"""

import torch



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
