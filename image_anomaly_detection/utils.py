"""
Provides utility functions for anomaly detection
"""

import torch
from torchvision import transforms as T
from PIL import Image
from score_calculation import calculate_patch_score, calculate_image_score, \
calculate_patch_classification, calculate_image_classification
from feature_extraction import extract_embedding_vectors



def anomaly_calculation(batch, features_model, mean, cov_inv, device, indices=None):
    embedding_vectors = extract_embedding_vectors(features_model, batch, device, indices=indices)
    patch_scores = calculate_patch_score(mean, cov_inv, embedding_vectors, device)
    image_scores = calculate_image_score(patch_scores)
    return patch_scores, image_scores


def anomaly_detection(batch, features_model, mean, cov_inv, device, thresh, indices=None):
    patch_scores, image_scores = anomaly_calculation(batch, features_model,
                                                     mean, cov_inv, device, indices=indices)
    patch_classifications = calculate_patch_classification(patch_scores, thresh)
    image_classifications = calculate_image_classification(image_scores, thresh)
    return patch_scores, image_scores, patch_classifications, image_classifications


def anomaly_detection_numpy(images, features_model, mean, cov_inv, device, thresh, indices=None):
    batch = to_batch(images, device)
    return anomaly_detection(batch, features_model, mean, cov_inv, device, thresh, indices)




def to_batch(images, device):
    cropsize = 224
    transform_x = T.Compose([
        T.Resize(cropsize),
        T.CenterCrop(cropsize),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    batch = torch.zeros((len(images), 3, cropsize, cropsize))
    for i, image in enumerate(images):
        image = Image.fromarray(image).convert('RGB')
        image = transform_x(image)
        batch[i] = image

    batch = batch.to(device)
    return batch
