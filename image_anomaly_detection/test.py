"""
Provides functions for testing trained models on images and whole datasets.
"""

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union, Tuple, Any
from .padim import Padim
from .patch_core import PatchCore


def eval_data(model: Union[Padim, PatchCore], dataloader: DataLoader) \
                -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on all images in dataloader and return images, labels and predictions."""

    images = []
    image_classifications_target = []
    masks_target = []
    image_scores = []
    score_maps = []

    for (batch, image_classifications, masks) in tqdm(dataloader, 'Inference'):
        batch_image_scores, batch_score_maps = model.predict(batch)

        images.extend(batch.cpu().numpy())
        image_classifications_target.extend(image_classifications.cpu().numpy())
        masks_target.extend(masks.cpu().numpy())
        image_scores.extend(batch_image_scores.cpu().numpy())
        score_maps.extend(batch_score_maps.cpu().numpy())

    return np.array(images), np.array(image_classifications_target), \
        np.array(masks_target).flatten().astype(np.uint8), \
        np.array(image_scores), np.array(score_maps).flatten()


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
