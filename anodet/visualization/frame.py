import numpy as np
import torch
import cv2
from typing import Union, Tuple
from .utils import to_numpy, frame_image


def frame_by_anomalies(images: Union[np.ndarray, torch.Tensor],
                       image_classifications: Union[np.ndarray, torch.Tensor],
                       padding: int = 30,
                       ano_color: Tuple[int, int, int] = (255, 0, 0),
                       non_ano_color: Tuple[int, int, int] = (0, 255, 0)
                       ) -> np.ndarray:

    """
        Frame images based on if anomaly is present in the image

       Args:
           images: Images on which to draw boundaries.
           image_classifications: information about, if the images have anomalies
           padding: the thickness of the border around the images.
           ano_color: color of frame if anomaly present.
           non_ano_color: color of frame if no anomaly present

       Returns:
           framed_images: Images with frames.

    """
    framed_images = []
    image_classifications = to_numpy(image_classifications).copy()
    images = to_numpy(images).copy()

    for i, image in enumerate(images):
        old_height, old_width = image.shape[:-1]

        if image_classifications[i]:
            f_image = frame_image(image, padding=padding, color=non_ano_color)
        else:
            f_image = frame_image(image, padding=padding, color=ano_color)

        f_image = cv2.resize(f_image, (old_width, old_height), interpolation=cv2.INTER_AREA)

        framed_images.append(f_image)

    return np.array(framed_images)
