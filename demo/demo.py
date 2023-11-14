import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/helvetica/_master_anodet/anodet')

import anodet

dataset_path = os.path.realpath("/Users/helvetica/_master_anodet/anodet/data_warehouse/dataset/")
distributions_path = os.path.realpath("/Users/helvetica/_master_anodet/anodet/data_warehouse/dataset/")

model = anodet.Padim(backbone="resnet18")
object_name = ["purple_duck"]
cam_names = ["cam_0_left", "cam_1_right"]


def get_dataloader(dataset_path, cam_names, object_name):
    dataset = anodet.AnodetDataset(
        os.path.join(dataset_path, f"{object_name}/train/good/{cam_names}")
    )
    dataloader = DataLoader(dataset, batch_size=32)
    print(
        f"DataLoader created with {len(dataset)} images for {object_name}, camera {cam_name}."
    )
    return dataloader


def model_fit(model, dataloader, distributions_path, cam_names, object_name):
    model.fit(dataloader)
    torch.save(
        model.mean,
        os.path.join(distributions_path, f"{object_name}_{cam_names}_mean.pt"),
    )
    torch.save(
        model.cov_inv,
        os.path.join(distributions_path, f"{object_name}_{cam_names}_cov_inv.pt"),
    )
    print(f"Parameters saved at {distributions_path}")


def predict(
    dataset_path, distributions_path, cam_names, object_name, test_images, THRESH=13
):
    images = []
    for path in test_images:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    batch = anodet.to_batch(
        images, anodet.standard_image_transform, torch.device("cpu")
    )

    mean = torch.load(
        os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_names}_mean.pt")
    )
    cov_inv = torch.load(
        os.path.join(distributions_path, f"{object_name}/{object_name}_{cam_names}_cov_inv.pt")
    )

    padim = anodet.Padim(
        backbone="resnet18", mean=mean, cov_inv=cov_inv, device=torch.device("cpu")
    )
    image_scores, score_maps = padim.predict(batch)

    score_map_classifications = anodet.classification(score_maps, THRESH)
    image_classifications = anodet.classification(image_scores, THRESH)

    test_images = np.array(images).copy()

    boundary_images = anodet.visualization.framed_boundary_images(test_images, score_map_classifications, image_classifications, padding=40)
    heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    highlighted_images = anodet.visualization.highlighted_images(images, score_map_classifications, color=(128, 0, 128))

    
    for idx in range(len(images)):
        fig, axs = plt.subplots(1, 4, figsize=(12, 6))
        fig.suptitle('Image: ' + str(idx), y=0.75, fontsize=14)
        axs[0].imshow(images[idx])
        axs[1].imshow(boundary_images[idx])
        axs[2].imshow(heatmap_images[idx])
        axs[3].imshow(highlighted_images[idx])
        plt.show()

    heatmap_images = anodet.visualization.heatmap_images(test_images, score_maps, alpha=0.5)
    tot_img = anodet.visualization.merge_images(heatmap_images, margin=40)
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    plt.imshow(tot_img)
    plt.show()

    return image_classifications, image_scores, score_maps

def visualization(distributions_path, test_image):

    # Test image
    image = cv2.imread(test_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load model
    mean = torch.load(distributions_path)      #TODO correct path implementatio
    cov_inv = torch.load(distributions_path)

    padim = anodet.Padim(backbone='resnet18', mean=mean, cov_inv=cov_inv, device=troch.device('cpu'))

    image_scores, score_maps = padim.predict(batch)



    return("adf")

if __name__ == "__main__":
    for cam_name in cam_names:
        # #  We won't train during demo.
        # dataloader = get_dataloader(dataset_path, cam_name, object_name[0])
        # model_fit(model, dataloader, distributions_path, cam_name, object_name[0])

        #  Replace with images taken during demo ⬇

        anomaly = ["Albinism"]  # Replace with camera output dir

        test_images = [
            os.path.join(
                dataset_path, f"{object_name[0]}/test/{anomaly[0]}/{cam_name}/000.png"
            ),
            # os.path.join(
            #     dataset_path, f"{object_name[0]}/test/{anomaly[0]}/{cam_name}/001.png"
            # ),
            # os.path.join(
            #     dataset_path, f"{object_name[0]}/test/{anomaly[0]}/{cam_name}/002.png"
            # ),
            # os.path.join(
            #     dataset_path, f"{object_name[0]}/test/good/{cam_name}/000.png"
            # ),
            # os.path.join(
            #     dataset_path, f"{object_name[0]}/test/good/{cam_name}/001.png"
            # ),
        ]

        #  For this we might wanna use mccp here ⬆

        results = predict(
            dataset_path, distributions_path, cam_name, object_name[0], test_images
        )

        for i in range(len(results[0])):

            image_file = cv2.imread(test_images[i])

            if results[0][i] == 0:

                print(
                    f"Image {i} from {cam_name}: 🦆 Anomaly duck detected: ({results[1][i]})."
                )

                #  Save heatmap only
                # plt.imshow(results[2][i], cmap='hot', interpolation='nearest')
                # plt.savefig(f"{cam_name}_heatmap_{i}.png")

                #  Save heatmap + original image overlay (#TODO)
                # heatmap_overlay = anodet.visualization.heatmap_image(image=image_file, patch_scores=results[2][i])
                # cv2.imwrite(f"{cam_name}_heatmap_{i}.png", heatmap_overlay)

                print(f"Image {i} from {cam_name}: 🦆 Anomaly duck detected: ({results[1][i]}).")

            else:
                print(f"Image {i} from {cam_name}: Passed. ({results[1][i]})")
