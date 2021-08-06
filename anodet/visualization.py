"""
Provides functions for visualizing results of anomaly detection.
"""

import cv2
import numpy as np
from skimage.segmentation import mark_boundaries
from typing import Union, Tuple, Optional, cast
import torch


def boundary_image(image: Union[np.ndarray, torch.Tensor],
                   mask: Union[np.ndarray, torch.Tensor],
                   color: Optional[Tuple[int, int, int]]=(255, 0, 0)
                  ):
    """
    Draw boundaries around masked areas on image.

    Args:
        image: Image on which to draw boundaries.
        mask: Mask defining the areas.
        color: Color of boundaries.

    Returns:
        boundary_image: Image with boundaries.

    """

    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    if type(mask) == torch.Tensor:
        mask = mask.cpu().numpy()

    image = cast(np.ndarray, image)
    mask = cast(np.ndarray, mask)

    boundary_image = image.copy()
    height, width, channels = boundary_image.shape

    # Draw boundaries
    boundaries = np.zeros(mask.shape)
    boundaries = mark_boundaries(boundaries, mask, color=(1, 0, 0), mode='thick')
    boundaries = cv2.resize(boundaries, (width, height), interpolation=cv2.INTER_AREA)
    boundaries = boundaries[:, :, 0].astype(bool)

    # Draw boundaries on image
    boundary_image[boundaries] = color

    return boundary_image


def heatmap_image(image: Union[np.ndarray, torch.Tensor],
                  score_map: Union[np.ndarray, torch.Tensor],
                  interval: Tuple[float, float]
                 ):
    """
    Draw heatmap on image based on score_map.

    Args:
        image: Image on which to the heatmap.
        score_map: Array which the heatmap is generated from.
        interval: Range of the heatmap (min_value, max_value)

    Returns:
        heatmap_image: Image with applied heatmap.

    """

    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    if type(score_map) == torch.Tensor:
        score_map = score_map.cpu().numpy()

    image = cast(np.ndarray, image)
    score_map = cast(np.ndarray, score_map)

    heatmap_image = image.copy()
    score_map = score_map.copy()
    height, width, channels = heatmap_image.shape
    min_value = interval[0]
    max_value = interval[1]

    # Clip
    score_map[score_map<min_value] = min_value
    score_map[score_map>=max_value] = max_value

    # Prepare heatmap
    score_map_norm = ((score_map - min_value) / max_value)
    score_map_norm_inv = ((1 - score_map_norm) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(score_map_norm_inv, colormap=cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_AREA)

    # Blend images
    heatmap_image = cv2.addWeighted(heatmap, 0.5, heatmap_image, 0.5, 0)

    return heatmap_image
