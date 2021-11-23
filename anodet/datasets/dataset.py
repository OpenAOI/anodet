import os
import torch
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from ..utils import standard_image_transform, standard_mask_transform


def allowed_file(filename):
    return ("." in filename and filename.rsplit(".", 1)[1].lower() in ["png", "jpg", "jpeg"])


class AnodetDataset(Dataset):

    def __init__(self, image_directory_path: str,
                 mask_directory_path: Optional[str] = None,
                 image_transforms: T.Compose = standard_image_transform,
                 mask_transforms: T.Compose = standard_mask_transform,
                 images_limit=None) -> None:

        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

        # Load image paths
        self.image_directory_path = image_directory_path
        self.image_paths = []
        for i, file in enumerate(os.listdir(self.image_directory_path)):
            if images_limit is not None and i >= images_limit:
                break
            filename = os.fsdecode(file)
            if allowed_file(filename):
                self.image_paths.append(os.path.join(self.image_directory_path, filename))

        # Load mask paths if mask_directory_path argument is given
        self.mask_directory_path = mask_directory_path
        self.mask_paths = []
        if self.mask_directory_path is not None:
            for file in os.listdir(self.mask_directory_path):
                filename = os.fsdecode(file)
                if allowed_file(filename):
                    self.mask_paths.append(os.path.join(self.mask_directory_path, filename))

            assert len(self.image_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
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
