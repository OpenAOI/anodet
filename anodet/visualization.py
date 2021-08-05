"""
Provides functions for visualizing results of anomaly detection.
"""

import cv2
import numpy as np
from skimage.segmentation import mark_boundaries



def boundary_image_classification_group(images, patch_classifications,
                                        image_classifications, size):

    padding = 30
    margin = 50
    tot_image = np.ones((size+2*padding, len(images)*(size+2*padding)\
                         +(len(images)-1)*margin, 3)).astype(np.uint8)*255

    for i, image in enumerate(images):
        b_image = boundary_image_classification(image, patch_classifications[i],
                                                image_classifications[i],
                                                size, padding=padding)
        height = b_image.shape[0]
        tot_image[:, i*(height+margin):(i+1)*(height)+i*margin, :] = b_image

    return tot_image



def boundary_image_classification(image, patch_classification,
                                  image_classification, size, padding=20):

    frame = (np.ones((size+2*padding, size+2*padding, 3))*255).astype(np.uint8)

    if image_classification:
        frame[:, :, 0] = 0
        frame[:, :, 2] = 0
    else:
        frame[:, :, 1] = 0
        frame[:, :, 2] = 0

    b_image = boundary_image(image, patch_classification, size)
    frame[padding:frame.shape[0]-padding, padding:frame.shape[1]-padding] = b_image
    return frame



def boundary_image(image, patch_classification, size):

    image = image.copy()
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    mask = patch_classification.cpu().numpy()

    transparent = np.zeros(mask.shape)
    line_img = mark_boundaries(transparent, mask, color=(1, 0, 0), mode='thick')
    line_img = cv2.resize(line_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    line_img = (line_img*255).astype(np.uint8)

    line_mask = line_img == [255, 0, 0]
    line_mask = line_mask[:, :, 0]

    image[:, :, 0] = np.where(line_mask, line_img[:, :, 0], image[:, :, 0])
    image[:, :, 1] = np.where(line_mask, 0, image[:, :, 1])
    image[:, :, 2] = np.where(line_mask, 0, image[:, :, 1])

    return image
