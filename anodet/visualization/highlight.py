import torch
import numpy as np
from .utils import to_numpy, blend_image
from typing import Union, Tuple


def highlighted_images(images: Union[np.ndarray, torch.Tensor],
                       patch_classifications: Union[np.ndarray, torch.Tensor],
                       color: Tuple[int, int, int] = (255, 0, 0),
                       alpha: float = 0.5
                       ) -> np.ndarray:
    """
    Highlights image areas that contains anomalies on multiple images.

    Args:
        images: The images to be drawn upon.
        patch_classifications: Anomaly classifications about the images.
        color: The highlight color.
        alpha: opacity of the highlight

    Returns:
        h_images: array of highlighted images.

    """
    images = to_numpy(images).copy()
    masks = to_numpy(patch_classifications).copy()
    h_images = []

    for i, image in enumerate(images):
        h_image = highlighted_image(image, masks[i], color=color, alpha=alpha)
        h_images.append(h_image)

    return np.array(h_images)


def highlighted_image(image: Union[np.ndarray, torch.Tensor],
                      patch_classification: Union[np.ndarray, torch.Tensor],
                      color: Tuple[int, int, int] = (255, 0, 0),
                      alpha: float = 0.5
                      ) -> np.ndarray:

    """
    Highlights image areas that contains anomalies

    Args:
        image: The image to be drawn upon.
        patch_classification: Anomaly classifications about the image.
        color: The highlight color.
        alpha: opacity of the highlight

    Returns:
        h_image: array of highlighted images.

    """
    image = to_numpy(image).copy()
    mask = to_numpy(patch_classification).copy()
    mask = np.logical_not(mask).astype(np.uint8)

    mask_height, mask_width = mask.shape
    mask_shape = (mask_height, mask_width, 3)
    mask_img = np.zeros(mask_shape, dtype=np.uint8)
    r, g, b = color

    mask_img[:, :, 0] = np.where(mask, r, 0)
    mask_img[:, :, 1] = np.where(mask, g, 0)
    mask_img[:, :, 2] = np.where(mask, b, 0)

    h_image = blend_image(image, mask_img, alpha=alpha, mask=mask)

    return np.array(h_image)
