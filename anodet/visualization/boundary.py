import numpy as np
from skimage.segmentation import find_boundaries
from .utils import frame_image, composite_image, to_numpy
from typing import Union, Tuple
import torch


def framed_boundary_images(images: Union[np.ndarray, torch.Tensor],
                           patch_classifications: Union[np.ndarray, torch.Tensor],
                           image_classifictions: Union[np.ndarray, torch.Tensor],
                           padding: int = 30,
                           boundary_color: Tuple[int, int, int] = (255, 0, 0)
                           ) -> np.ndarray:

    """
       Draw boundaries around masked areas on images and adds
       a frame around the image that indicates if a boundary was drawn.

       Args:
           images: Images on which to draw boundaries.
           patch_classifications: anomaly classifcations about the images.
           image_classifictions: information about, if the images have anomalys
           padding: the thickness of the border around the images.
           boundary_color: Color of boundaries.

       Returns:
           b_image: Image with boundaries.

    """
    result_images = []

    images = to_numpy(images)
    masks = to_numpy(patch_classifications)
    image_classifictions = to_numpy(image_classifictions)

    for i, image in enumerate(images):
        b_image = boundary_image(image, masks[i], boundary_color=boundary_color)

        if image_classifictions[i]:
            b_image = frame_image(b_image, padding=padding, color=(255, 0, 0))
        else:
            b_image = frame_image(b_image, padding=padding, color=(0, 255, 0))

        result_images.append(b_image)

    return np.array(result_images)


def boundary_image(image: Union[np.ndarray, torch.Tensor],
                   patch_classification: Union[np.ndarray, torch.Tensor],
                   boundary_color: Tuple[int, int, int] = (255, 0, 0)
                   ) -> np.ndarray:
    """
       Draw boundaries around masked areas on image.

       Args:
           image: Image on which to draw boundaries.
           patch_classification: Mask defining the areas.
           boundary_color: Color of boundaries.

       Returns:
           b_image: Image with boundaries.

    """

    image = to_numpy(image)
    mask = to_numpy(patch_classification)
    image = image.copy()

    found_boundaries = find_boundaries(mask).astype(np.uint8)
    found_boundaries = np.logical_not(found_boundaries).astype(np.uint8)
    layer_two = np.zeros(image.shape, dtype=np.uint8)
    layer_two[:] = boundary_color

    b_image = composite_image(image, layer_two, found_boundaries)

    return b_image
