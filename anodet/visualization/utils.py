import numpy as np
import torch
from PIL import Image
import cv2
from typing import Union, Tuple, cast, Optional


def merge_images(images: Union[np.ndarray, torch.Tensor],
                 margin: int = 0
                 ) -> np.ndarray:
    """
    Combine multiple images to a big image, with a specified margin,
    between the images.
    Args:
        images: The images to merge
        margin: Margin between the images

    Returns:
        tot_image: The merged image
    """
    images = to_numpy(images)
    amount, height, width, channels = images.shape
    tot_image_height = height
    tot_image_width = amount * width + (amount - 1) * margin
    tot_image_size = (tot_image_height, tot_image_width, channels)
    tot_image = np.ones(tot_image_size).astype(np.uint8) * 255

    for i, image in enumerate(images):
        start_x = i * (margin + width)
        end_x = (i + 1) * width + (i * margin)
        tot_image[:, start_x:end_x, :] = image

    return tot_image


def frame_image(image: Union[np.ndarray, torch.Tensor],
                padding: int = 30,
                color: Tuple[int, int, int] = (0, 0, 0)
                ) -> np.ndarray:
    """
    Draws a colored frame around a image.
    Args:
        image: The image to put frame around.
        padding: The thickness of the frame.
        color: The color of the frame.

    Returns:
        f_image: The framed image.
    """
    image = to_numpy(image)
    height, width, channels = image.shape
    image_height = height + 2 * padding
    image_width = width + 2 * padding
    image_shape = (image_height, image_width, channels)
    f_image = np.ones(image_shape, dtype=np.uint8)
    f_image[:] = color
    f_image[padding:image_height - padding, padding: image_width - padding] = image

    return f_image


def blend_image(image_one: Union[np.ndarray, torch.Tensor],
                image_two: Union[np.ndarray, torch.Tensor],
                alpha: float = 0.5
                ) -> np.ndarray:
    """
    Draws image on another image, with a set opacity.
    Args:
        image_one: The base image.
        image_two: The image to draw with.
        alpha: The opacity of image two.

    Returns:
        blended_image: The blended image.

    """
    image_one = to_numpy(image_one)
    image_two = to_numpy(image_two)
    height, width, channels = image_one.shape
    layer_one = image_one.copy()
    layer_two = image_two.copy()

    layer_two = cv2.resize(layer_two, (height, width), interpolation=cv2.INTER_AREA)
    blended_image = Image.blend(
        Image.fromarray(layer_one),
        Image.fromarray(layer_two),
        alpha=alpha
    )

    return np.array(blended_image)


def composite_image(image_one: Union[np.ndarray, torch.Tensor],
                    image_two: Union[np.ndarray, torch.Tensor],
                    mask: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Draws image_two over image_one using a mask,
    Areas marked a
s 1 is transparent and 0 draws image_two with
    opacity 1.

    Args:
        image_one: The base image.
        image_two: The image to draw with.
        mask: mask on where to draw the image.

    Returns:
        tot_Image: The combined image.

    """
    image_one = to_numpy(image_one)
    image_two = to_numpy(image_two)
    mask = to_numpy(mask)

    height, width, channels = image_one.shape

    image_two = cv2.resize(image_two, (height, width), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (height, width), interpolation=cv2.INTER_AREA)
    mask = np.array(mask).astype(bool)
    mask = Image.fromarray(mask)
    image_one = Image.fromarray(image_one)
    image_two = Image.fromarray(image_two)
    tot_image = Image.composite(image_one, image_two, mask)

    return np.array(tot_image)


def normalize_patch_scores(patch_scores: Union[np.ndarray, torch.Tensor],
                           min_v: Optional[float] = None,
                           max_v: Optional[float] = None,
                           ) -> np.ndarray:
    """
    Takes a set of patch_scores and normalize them to values between 0-1.
    Args:
        max_v:
        min_v:
        patch_scores: array of patch_scores.

    Returns:
        normalized_matrix: A normalized numpy array.
    """
    patch_scores = to_numpy(patch_scores)

    if min_v and max_v:
        min_score = min_v
        max_score = max_v
        patch_scores[patch_scores < min_score] = min_score
        patch_scores[patch_scores >= max_score] = max_score
    else:
        min_score = patch_scores.min()
        max_score = patch_scores.max()

    normalized_matrix = normalize_matrix(patch_scores, min_score, max_score)
    return normalized_matrix


def normalize_matrix(matrix: np.ndarray,
                     minimum: float,
                     maximum: float) -> np.ndarray:
    """
    Args:
        matrix: Matrix to be normalized.
        minimum: minimum value to use in normalization.
        maximum: maximum value to use in normalization.

    Returns:
        normalized matrix
    """
    return (matrix - minimum) / (maximum - minimum)


def to_numpy(in_array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Casts a tensor to a numpy array

    Args:
        in_array: The array to be casted

    Returns:
        np_array: a casted numpy array

    """
    if isinstance(in_array, torch.Tensor):
        in_array = in_array.cpu().numpy()

    np_array = cast(np.ndarray, in_array)

    return np_array
