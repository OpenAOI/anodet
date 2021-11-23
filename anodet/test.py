"""
Provides functions for testing trained models on images and whole datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from typing import Tuple, Any
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

import os
import time
from time import gmtime
from time import strftime

import anodet


def visualize_eval_data(image_classifications_target: np.ndarray, masks_target:
                        np.ndarray, image_scores: np.ndarray, score_maps: np.ndarray) -> None:
    """Visualize image and pixel level results from eval_data."""

    print("Image level")
    visualize_eval_pair(image_classifications_target, image_scores)
    print("Pixel level")
    visualize_eval_pair(masks_target, score_maps)


def visualize_eval_pair(target: np.ndarray, prediction: np.ndarray) -> None:
    """Visualize results of binary prediction."""

    score = roc_auc_score(target, prediction)
    print('ROC-AUC score:', score)
    print()

    precision, recall, threshold = optimal_threshold(target, prediction)
    print('Optimal thresh:', threshold)
    print('Recall:', recall)
    print('Precision:', precision)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

    fpr, tpr, thresholds = roc_curve(target, prediction)
    axes[0].plot(fpr, tpr)
    axes[0].title.set_text('ROC Curve (tpr-fpr)')

    axes[1].plot(thresholds, fpr)
    axes[1].plot(thresholds, tpr, color='red')
    axes[1].axvline(x=threshold, color='yellow')
    axes[1].grid()
    axes[1].title.set_text('fpr/tpr - thresh')

    plt.show()


def optimal_threshold(target: np.ndarray, prediction: np.ndarray) -> Tuple[Any, Any, Any]:
    """Calculate optimal threshold for binary prediction."""

    precision, recall, thresholds = precision_recall_curve(target.flatten(), prediction.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    idx = np.argmax(f1)
    return precision[idx], recall[idx], thresholds[idx]


def run_padim_test(backbone: str = 'resnet34',
                     tresh: int = 13,
                     extractions: int = 1,
                     batch_size: int = 32,
                     train_images_limit=None,
                     test_good_images_limit=None,
                     test_bad_images_limit=None,
                     image_transforms: T.Compose = T.Compose([T.Resize(224),
                                                              T.CenterCrop(224),
                                                              T.ToTensor(),
                                                              T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])
                                                              ])
                     ):
    """
    Script to run Padim model using AnodetDataset.

        Args:
            backbone: The name of the desired backbone. Must be one of: [resnet18, resnet34 \
            wide_resnet50]
            tresh: A treshold value. If an image score is larger than
                or equal to thresh it is classified as anomalous.
            extractions: Number of extractions from dataloader. Could be of interest \
                when applying random augmentations.
            batch_size: Size of batch for dataloader.
            train_images_limit: Limit number of images to train on model.
            test_good_images_limit: Limit number of good images to test on model.
            test_bad_images_limit: Limit number of bad images to test on model.
            image_transforms: Torchvision Compose obj containing transforms.

        Returns:
            image_classification_target: Numpy array with target classification.
            image_classifications: Numpy array with predicted image classifcations.

    """
    # Paths
    dataset_path = os.path.realpath('../../data/pscb/good_cropped_masked')
    test_good_images_path = os.path.realpath('../../data/pscb/test_good_cropped_masked')
    test_bad_images_path = os.path.realpath('../../data/pscb/bad_cropped_masked')

    # Start time
    start_time = time.time()

    # Load dataset
    dataset = anodet.AnodetDataset(image_directory_path=dataset_path,
                                   image_transforms=image_transforms,
                                   images_limit=train_images_limit)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Init the model
    model = anodet.Padim(backbone=backbone)
    model.fit(dataloader=dataloader, extractions=extractions)

    # Load good and bad test images
    good_image_paths = anodet.get_paths_for_directory_path(
        directory_path=test_good_images_path,
        limit=test_good_images_limit)
    bad_image_paths = anodet.get_paths_for_directory_path(
        directory_path=test_bad_images_path,
        limit=test_bad_images_limit)
    test_paths = good_image_paths + bad_image_paths

    # Target array image level
    image_classification_target = [1] * len(good_image_paths) + [0] * len(bad_image_paths)
    image_classification_target = np.array(image_classification_target)

    images = []
    for path in test_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    batch = anodet.to_batch(images, image_transforms, torch.device('cpu'))

    # Make prediction
    image_scores, score_maps = model.predict(batch)

    # Interpret the prediction
    score_map_classifications = anodet.classification(score_maps, tresh)
    image_classifications = anodet.classification(image_scores, tresh)
    image_classifications = image_classifications.numpy()

    # Calculate time
    end_time = time.time() - start_time
    end_time = strftime("%H:%M:%S", gmtime(end_time))

    # Print score and arguments
    score_roc_auc = roc_auc_score(image_classification_target,
                                  image_classifications)
    precision, recall, thresholds = anodet.optimal_threshold(image_classification_target,
                                                             image_classifications)
    result = {"roc_auc_score": score_roc_auc, "optimal_trest": thresholds, "precision": precision, "recall":
              recall, "time": end_time, "backbone": backbone, "tresh": tresh, "extractions": extractions,
              "train_images_limit": train_images_limit, "image_transforms": str(image_transforms)}
    print("Padim script completed with results: ", result)

    return image_classification_target, image_classifications
