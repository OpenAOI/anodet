"""
Padim Script
"""

import os
import anodet
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms as T


def visual(test_images, score_map_classifications, image_classifications, score_maps, images, image_scores):
    boundary_images = anodet.visualization.framed_boundary_images(test_images, score_map_classifications,
                                                                  image_classifications, padding=40)
    heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    highlighted_images = anodet.visualization.highlighted_images(images, score_map_classifications, color=(128, 0, 128))

    for i, idx in enumerate(range(len(images))):
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        fig.suptitle('Image: ' + str(idx) + " score: " + str(image_scores[i].item()), y=0.75, fontsize=20)
        axs[0].imshow(images[idx])
        axs[1].imshow(boundary_images[idx])
        axs[2].imshow(heatmap_images[idx])
        axs[3].imshow(highlighted_images[idx])
        plt.show()


def run(dataset_path: str = '../../data/pscb/good_cropped_masked',
        test_images_path: str = '../../data/pscb/test_good_cropped_masked',
        model_data_path: str = './distributions/',
        backbone: str = 'resnet18',
        layer_indices=None,
        tresh: int = 13,
        extractions: int = 1,
        test_images_limit=10,
        image_transforms: T.Compose = T.Compose([T.Resize(224),
                                                 T.CenterCrop(224),
                                                 T.ToTensor(),
                                                 T.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
                                                 ])
        ):
    # Load dataset
    dataset = anodet.AnodetDataset(image_directory_path=dataset_path,
                                   image_transforms=image_transforms)
    dataloader = DataLoader(dataset, batch_size=32)

    # Init the model
    padim = anodet.Padim(backbone=backbone)
    padim.fit(dataloader=dataloader, extractions=extractions)

    # Save the necessary parameters
    distributions_path = './distributions/'
    torch.save(padim.mean, os.path.join(model_data_path, "pscb_mean.pt"))
    torch.save(padim.cov_inv, os.path.join(model_data_path, "pscb_inv.pt"))

    # Load test images
    test_paths = anodet.get_paths_for_directory_path(
        directory_path=test_images_path,
        limit=test_images_limit)

    images = []
    for path in test_paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    batch = anodet.to_batch(images, image_transforms, torch.device('cpu'))

    # Load the model data
    mean = torch.load(os.path.join(model_data_path, 'pscb_mean.pt'))
    cov_inv = torch.load(os.path.join(model_data_path, 'pscb_inv.pt'))

    # Init the model
    padim = anodet.Padim(backbone=backbone, mean=mean, cov_inv=cov_inv, device=torch.device('cpu'))

    # Make prediction
    image_scores, score_maps = padim.predict(batch)

    # Interpret the prediction
    score_map_classifications = anodet.classification(score_maps, tresh)
    image_classifications = anodet.classification(image_scores, tresh)

    # Visualization
    test_images = np.array(images).copy()

    # Run visualization... TODO: Save to file
    visual(test_images, score_map_classifications, image_classifications, score_maps, images, image_scores)


run()
