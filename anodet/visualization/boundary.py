import numpy as np
from skimage.segmentation import find_boundaries
from .utils import composite_image, to_numpy
from typing import Union, Tuple
from .frame import frame_by_anomalies
import torch


def framed_boundary_images(images: Union[np.ndarray, torch.Tensor],
                           patch_classifications: Union[np.ndarray, torch.Tensor],
                           image_classifications: Union[np.ndarray, torch.Tensor],
                           padding: int = 30,
                           boundary_color: Tuple[int, int, int] = (255, 0, 0)
                           ) -> np.ndarray:

    """
       Draw boundaries around masked areas on images and adds
       a frame around the image that indicates if a boundary was drawn.

       Args:
           images: Images on which to draw boundaries.
           patch_classifications: anomaly classifications about the images.
           image_classifications: information about, if the images have anomalies
           padding: the thickness of the border around the images.
           boundary_color: Color of boundaries.

       Returns:
           b_image: Image with boundaries.

    """

    images = to_numpy(images).copy()
    masks = to_numpy(patch_classifications).copy()
    image_classifications = to_numpy(image_classifications).copy()

    b_images = boundary_images(images, masks, boundary_color=boundary_color)
    framed_b_images = frame_by_anomalies(
        b_images,
        image_classifications,
        padding=padding
    )

    return np.array(framed_b_images)


def boundary_images(images: Union[np.ndarray, torch.Tensor],
                    patch_classifications: Union[np.ndarray, torch.Tensor],
                    boundary_color: Tuple[int, int, int] = (255, 0, 0)
                    ) -> np.ndarray:
    """
       Draw boundaries around masked areas on images and adds
       a frame around the image that indicates if a boundary was drawn.

       Args:
           images: Images on which to draw boundaries.
           patch_classifications: anomaly classifications about the images.
           boundary_color: Color of boundaries.

       Returns:
           b_image: Image with boundaries.

    """

    images = to_numpy(images).copy()
    masks = to_numpy(patch_classifications).copy()
    images = [boundary_image(image, masks[i], boundary_color=boundary_color)
              for i, image in enumerate(images)]

    return np.array(images)


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

    image = to_numpy(image).copy()
    mask = to_numpy(patch_classification).copy()

    found_boundaries = find_boundaries(mask).astype(np.uint8)
    layer_two = np.zeros(image.shape, dtype=np.uint8)
    layer_two[:] = boundary_color

    b_image = composite_image(image, layer_two, found_boundaries)

    return b_image
