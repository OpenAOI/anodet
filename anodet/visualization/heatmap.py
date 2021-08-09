import cv2
import torch

from .utils import (normalize_patch_scores, blend_image)
import numpy as np
from typing import Union, Optional


def heatmap_images(images: Union[np.ndarray, torch.Tensor],
                   list_of_patch_scores: Union[np.ndarray, torch.Tensor],
                   min_v: Optional[float] = None,
                   max_v: Optional[float] = None,
                   alpha: float = 0.6) -> np.ndarray:
    """
    Takes array of images and patch_scores to create heatmaps on the images.

    Args:
        images: The images to draw heatmaps on.
        list_of_patch_scores: The values to use to generate colormap.
        min_v:
        max_v:
        alpha: The opacity of the colormap

    Returns:
        heatmaps: a array of heatmaps.

    """
    heatmaps = []
    norm_patch_scores = normalize_patch_scores(
        list_of_patch_scores,
        min_v=min_v,
        max_v=max_v
    )
    for i, score in enumerate(norm_patch_scores):
        image_heatmap = heatmap_image(images[i], score, alpha=alpha)
        heatmaps.append(image_heatmap)

    return np.array(heatmaps)


def heatmap_image(image: Union[np.ndarray, torch.Tensor],
                  norm_patch_scores: Union[np.ndarray, torch.Tensor],
                  alpha: float = 0.6) -> np.ndarray:
    """
    draws a heatmap over a image using patch_scores to
    indicate areas of interest.

    Args:
        image: image to draw the colormap on.
        norm_patch_scores: normalized patch scores
        alpha: Opacity on the colormap

    Returns:
        heatmap: Combination of image and colormap.


    """
    norm_patch_scores = (1 - norm_patch_scores) * 255
    norm_patch_scores = norm_patch_scores.astype(np.uint8)
    color_map = cv2.applyColorMap(norm_patch_scores, colormap=cv2.COLORMAP_JET)
    heatmap = blend_image(image, color_map, alpha=alpha)
    return heatmap
