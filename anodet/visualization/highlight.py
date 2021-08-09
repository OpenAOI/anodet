from .utils import to_numpy, blend_image
from typing import Union, Tuple
import torch
import numpy as np


def highlighted_images(images: Union[np.ndarray, torch.Tensor],
                       patch_classifications: Union[np.ndarray, torch.Tensor],
                       color: Tuple[int, int, int] = (255, 0, 0)
                       ) -> np.ndarray:
    """
    Highlights imageareas that contains anomalys on multiple images.

    Args:
        images: The images to be drawn upon.
        patch_classifications: Anomaly classifcations about the images.
        color: The highlight color.

    Returns:
        h_images: array of higlighted images.

    """
    h_images = []
    for i, image in enumerate(images):
        masks = to_numpy(patch_classifications)
        h_image = highlighted_image(image, masks[i], color=color)
        h_images.append(h_image)

    return np.array(h_images)


def highlighted_image(image: Union[np.ndarray, torch.Tensor],
                      patch_classification: Union[np.ndarray, torch.Tensor],
                      color: Tuple[int, int, int] = (255, 0, 0)
                      ) -> np.ndarray:

    """
    Highlights imageareas that contains anomalys

    Args:
        image: The image to be drawn upon.
        patch_classification: Anomaly classifcations about the image.
        color: The highlight color.

    Returns:
        h_image: array of higlighted images.

    """

    mask = to_numpy(patch_classification)
    mask_height, mask_width = mask.shape
    mask_shape = (mask_height, mask_width, 3)
    mask_img = np.zeros(mask_shape, dtype=np.uint8)
    r, g, b = color

    mask_img[:, :, 0] = np.where(mask, 0, r)
    mask_img[:, :, 1] = np.where(mask, 0, g)
    mask_img[:, :, 2] = np.where(mask, 0, b)
    h_image = blend_image(image, mask_img)

    return np.array(h_image)
