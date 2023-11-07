import os, math
from typing import Callable, List, Optional, Tuple, Union 
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.utils.data import DataLoader

import numpy as np

from utils import standard_mask_transform
from feature_extraction import ResnetEmbeddingsExtractor
from padim import Padim
from utils import mahalanobis, pytorch_cov, split_tensor_and_run_function


standard_image_transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[1.485, 1.456, 1.406], std=[1.229, 1.224, 1.225]),
    ]
)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in [
        "png",
        "jpg",
        "jpeg",]

class AnodetDataset(Dataset):
    def __init__(
        self,
        image_directory_path: str,
        mask_directory_path: Optional[str] = None,
        image_transforms: T.Compose = standard_image_transform,
        mask_transforms: T.Compose = standard_mask_transform,
    ) -> None:
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

        # Load image paths
        self.image_directory_path = image_directory_path
        self.image_paths = []
        for file in os.listdir(self.image_directory_path):
            filename = os.fsdecode(file)
            if allowed_file(filename):
                self.image_paths.append(
                    os.path.join(self.image_directory_path, filename)
                )

        # Load mask paths if mask_directory_path argument is given
        self.mask_directory_path = mask_directory_path
        self.mask_paths = []
        if self.mask_directory_path is not None:
            for file in os.listdir(self.mask_directory_path):
                filename = os.fsdecode(file)
                if allowed_file(filename):
                    self.mask_paths.append(
                        os.path.join(self.mask_directory_path, filename)
                    )

            assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.image_transforms(image)

        # Load mask if mask_directory_path argument is given
        if self.mask_directory_path is not None:
            mask = Image.open(self.mask_paths[idx])
            mask = self.mask_transforms(mask)
            image_classification = 0
        else:
            mask = torch.zeros([1, image.shape[1], image.shape[2]])
            image_classification = 1

        return image, image_classification, mask


# Count rows and columns in saved tensor (Should be 224 x 224)
def get_rows_cols_from_txt(file_path):
    with open(file_path, 'r') as file:
        rows = sum(1 for line in file)
        file.seek(0)
        columns = len(file.readline().split())


        file_stats = os.stat(file_path)
        print(file_path)
        print(file)
        #print(file_stats)
        print(f"MB: {file_stats.st_size / (1024 * 1024)} \n")


    return rows, columns

def summary_stats_from_txt(file_path):
    with open(file_path) as f:
        data = [float(num) for line in f for num in line.split()]

    print("Minimum value:", min(data))
    print("Maximum value:", max(data))
    print("Average value:", sum(data) / len(data))


tensor_0 = '/Users/helvetica/Desktop/duck_rgb/arrays/tensor_image_0.txt'
tensor_1 = '/Users/helvetica/Desktop/duck_rgb/arrays/tensor_image_1.txt'
tensor_2 = '/Users/helvetica/Desktop/duck_rgb/arrays/tensor_image_2.txt'


if __name__ == "__main__":

    # Import image, parse through AnodetDataset
    original_duck = '/Users/helvetica/Desktop/duck_rgb/dir'
    duck_in_dataset = AnodetDataset(original_duck)

    # Parse dataset into DataLoader (torch.data)
    dloader = DataLoader(duck_in_dataset)

    # Print and save the duck-tensor
    for batch_idx, batch in enumerate(dloader):
        print(f"Batch {batch_idx}:")
        print(f"Batch content: \n {batch}")

        file_path = f"batch_{batch_idx}.pt"
        torch.save(batch, file_path)
        print(f"Batch saved as {file_path}")

    print("Loading the saved batch:")
    loaded_data = torch.load("batch_0.pt")
    print(loaded_data)

    # Get total tensors in batch (Should be 3)
    print(len(loaded_data))

    selected_tensor = loaded_data[0]

    print(f"Tensor size: {selected_tensor.size()}")
    print(f"Tensor shape: {selected_tensor.shape}")

    tensor_min = torch.min(selected_tensor)
    tensor_max = torch.max(selected_tensor)
    tensor_mean = torch.mean(selected_tensor)

    print(f"Minimum value: {tensor_min.item()}")
    print(f"Maximum value: {tensor_max.item()}")
    print(f"Average value: {tensor_mean.item()}")

    # rows, columns = get_rows_cols_from_txt(selected_tensor)
    # print(f"Rows   : {rows}")
    # print(f"Columns: {columns}")

    # summary_stats_from_txt(selected_tensor)


    # print(len(dataloader))

    model = Padim()
    print(model)
    print(model.channel_indices.shape)
    model.fit(dloader)  # Why division by 0 err???