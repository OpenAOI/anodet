import numpy as np
import torch
import cv2
from PIL import Image
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
    images = to_numpy(images).copy()

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
    image = to_numpy(image).copy()

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
                alpha: float = 0.5,
                mask: Optional[np.ndarray] = None
                ) -> np.ndarray:
    """
    Draws image on another image, with a set opacity.
    Args:
        image_one: The base image.
        image_two: The image to draw with.
        alpha: The opacity of image two.
        mask: 1 or True what parts of image_two that will be pasted n image_one

    Returns:
        blended_image: The blended image.

    """
    layer_one = to_numpy(image_one).copy()
    layer_two = to_numpy(image_two).copy()
    height, width, channels = layer_one.shape

    layer_two = cv2.resize(layer_two, (width, height), interpolation=cv2.INTER_AREA)

    blended_image = Image.blend(
        Image.fromarray(layer_one),
        Image.fromarray(layer_two),
        alpha=alpha
    )

    if isinstance(mask, (np.ndarray, torch.Tensor)):
        mask = to_numpy(mask).copy()
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)
        blended_image = composite_image(layer_one, np.array(blended_image), mask)  # type: ignore

    return np.array(blended_image)


def composite_image(image_one: Union[np.ndarray, torch.Tensor],
                    image_two: Union[np.ndarray, torch.Tensor],
                    mask: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Draws image_two over image_one using a mask,
    Areas marked as 1 is transparent and 0 draws image_two with
    opacity 1.

    Args:
        image_one: The base image.
        image_two: The image to draw with.
        mask: mask on where to draw the image.

    Returns:
        tot_Image: The combined image.

    """
    image_one = to_numpy(image_one).copy()
    image_two = to_numpy(image_two).copy()
    mask = to_numpy(mask).copy()

    height, width, channels = image_one.shape

    image_two = cv2.resize(image_two, (width, height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)

    mask = (mask == (1 | True))
    image_one[mask] = image_two[mask]

    return image_one


def normalize_patch_scores(patch_scores: Union[np.ndarray, torch.Tensor],
                           min_v: Optional[float] = None,
                           max_v: Optional[float] = None,
                           ) -> np.ndarray:
    """
    Takes a set of patch_scores and normalize them to values between 0-1.
    Args:
        max_v: max value for normalization
        min_v: min value for normalization
        patch_scores: array of patch_scores.

    Returns:
        normalized_matrix: A normalized numpy array.
    """
    patch_scores = to_numpy(patch_scores).copy()

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
