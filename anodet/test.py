"""
Provides functions for testing trained models on images and whole datasets.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from typing import Tuple, Any

def visualize_eval_data(image_classifications_target: np.ndarray, masks_target:
                        np.ndarray, image_scores: np.ndarray, score_maps: np.ndarray) -> None:
    """Visualize image and pixel level results from eval_data."""

    visualize_eval_pair(image_classifications_target, image_scores)
    # print("Pixel level")
    # visualize_eval_pair(masks_target, score_maps)


def visualize_eval_pair(target: np.ndarray, prediction: np.ndarray) -> None:
    """Visualize results of binary prediction."""

    score = roc_auc_score(target, prediction)
    best_precision, best_recall, best_f1, best_threshold = optimal_threshold(target, prediction)

    print('ROC-AUC score:', score)
    # print('PR-AUC socre:', prauc)
    print('Best Precision:', best_precision)
    print('Best Recall:', best_recall)
    print('Best f1 score: ', best_f1)
    print('Optimal thresh:', best_threshold)

    # PrecisionRecallDisplay.from_predictions(target.flatten(), prediction.flatten(), name="Precision-Recall")



    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))
    #
    # fpr, tpr, thresholds = roc_curve(target, prediction)
    # axes[0].plot(fpr, tpr)
    # axes[0].title.set_text('ROC Curve (tpr-fpr)')
    #
    # axes[1].plot(thresholds, fpr)
    # axes[1].plot(thresholds, tpr, color='red')
    # axes[1].axvline(x=best_threshold, color='yellow')
    # axes[1].grid()
    # axes[1].title.set_text('fpr/tpr - thresh')
    #
    plt.show()

def optimal_threshold(target: np.ndarray, prediction: np.ndarray) -> Tuple[Any, Any, Any, Any]:
    """Calculate optimal threshold for binary prediction."""

    precision, recall, thresholds = precision_recall_curve(target.flatten(), prediction.flatten())

    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    idx = np.argmax(f1)
    return precision[idx], recall[idx], f1[idx], thresholds[idx]



