import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/helvetica/_master_anodet/anodet')

from anodet import Padim, AnodetDataset, to_batch, classification, visualization, standard_image_transform

# Set dataset and distributions paths
dataset_path = os.path.realpath("/Users/helvetica/_master_anodet/anodet/data_warehouse/dataset/")
distributions_path = os.path.realpath("/Users/helvetica/_master_anodet/anodet/data_warehouse/dataset/")

# Initialize Padim model
model = Padim(backbone="resnet18")

# Define object and camera names
object_name = "purple_duck"
cam_names = ["cam_0_left", "cam_1_right"]

def get_dataloader(dataset_path, cam_name, object_name):
    dataset = AnodetDataset(os.path.join(dataset_path, f"{object_name}/train/good/{cam_name}"))
    dataloader = DataLoader(dataset, batch_size=32)
    print(f"DataLoader created with {len(dataset)} images for {object_name}, camera {cam_name}.")
    return dataloader

def model_fit(model, dataloader, distributions_path, cam_name, object_name):
    model.fit(dataloader)
    torch.save(model.mean, os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_name}_mean.pt"))
    torch.save(model.cov_inv, os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_name}_cov_inv.pt"))
    print(f"Parameters saved at {distributions_path}")

def predict(dataset_path, distributions_path, cam_name, object_name, test_images, THRESH=13):
    images = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in test_images]
    batch = to_batch(images, standard_image_transform, torch.device("cpu"))

    mean = torch.load(os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_name}_mean.pt"))
    cov_inv = torch.load(os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_name}_cov_inv.pt"))

    padim = Padim(backbone="resnet18", mean=mean, cov_inv=cov_inv, device=torch.device("cpu"))
    image_scores, score_maps = padim.predict(batch)

    score_map_classifications = classification(score_maps, THRESH)
    image_classifications = classification(image_scores, THRESH)

    test_images = np.array(images).copy()

    boundary_images = visualization.framed_boundary_images(test_images, score_map_classifications, image_classifications, padding=40)
    heatmap_images = visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    highlighted_images = visualization.highlighted_images(images, score_map_classifications, color=(128, 0, 128))

    for idx in range(len(images)):
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        fig.suptitle(f"Image: {idx}", y=0.75, fontsize=14)
        axs[0].imshow(images[idx])
        axs[1].imshow(boundary_images[idx])
        axs[2].imshow(heatmap_images[idx])
        axs[3].imshow(highlighted_images[idx])
        plt.show()

    heatmap_images = visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    tot_img = visualization.merge_images(heatmap_images, margin=40)
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    plt.imshow(tot_img)
    plt.show()

    return image_classifications, image_scores, score_maps

def main():
    for cam_name in cam_names:
        # Uncomment the following lines if you want to train the model during the demo
        # dataloader = get_dataloader(dataset_path, cam_name, object_name)
        # model_fit(model, dataloader, distributions_path, cam_name, object_name)

        # Replace with images taken during demo
        anomaly = ["Albinism"]  # Replace with camera output dir
        test_images = [os.path.join(dataset_path, f"{object_name}/test/{anomaly[0]}/{cam_name}/000.png")]

        results = predict(dataset_path, distributions_path, cam_name, object_name, test_images)

        for i, result in enumerate(zip(*results)):
            image_file = cv2.imread(test_images[i])
            image_classification, image_score, score_map = result

            if image_classification == 0:
                print(f"Image {i} from {cam_name}: 🦆 Anomaly duck detected: ({image_score}).")
            else:
                print(f"Image {i} from {cam_name}: Passed. ({image_score})")

if __name__ == "__main__":
    main()
