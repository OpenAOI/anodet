import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.abspath(os.path.join(dir_path+'/src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
    

from padim_utils import toBatch
from feature_extraction import Resnet18Features, WideResnet50Features, extractEmbeddingVectors, getOriginalResnet18Indicies
from score_calculation import calculatePatchScore, calculateImageScore, calculatePatchClassification, calculateImageClassification
from visualization import getBoundaryImage, getBoundaryImageClassification, getBoundaryImageClassificationGroup



def anomalyCalculation(batch, features_model, mean, cov_inv, device, indices=None):
    embedding_vectors = extractEmbeddingVectors(features_model, batch, device, indices=indices)
    patch_scores = calculatePatchScore(mean, cov_inv, embedding_vectors, device)
    image_scores = calculateImageScore(patch_scores)
    return patch_scores, image_scores


def anomalyDetection(batch, features_model, mean, cov_inv, device, thresh, indices=None):
    patch_scores, image_scores = anomalyCalculation(batch, features_model, mean, cov_inv, device, indices=indices)
    patch_classifications = calculatePatchClassification(patch_scores, thresh)
    image_classifications = calculateImageClassification(image_scores, thresh)
    return patch_scores, image_scores, patch_classifications, image_classifications


def anomalyDetectionNumpy(images, features_model, mean, cov_inv, device, thresh, indices=None):
    batch = toBatch(images, device)
    return anomalyDetection(batch, features_model, mean, cov_inv, device, thresh, indices)