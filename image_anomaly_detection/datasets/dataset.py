import os
from typing import Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T



def allowed_file(filename):
    return ("." in filename and filename.rsplit(".", 1)[1].lower() in ["png", "jpg", "jpeg"])


class Dataset(Dataset):

    def __init__(self, image_directory_path: str,
                 mask_directory_path: Optional[str] = None,
                 image_transforms: Optional[T.Compose] = None,
                 mask_transforms: Optional[T.Compose] = None) -> None:

        # Set image transfroms
        self.image_transforms = image_transforms
        if self.image_transforms is None:
            self.image_transforms = T.Compose([T.Resize(224),
                                               T.CenterCrop(224),
                                               T.ToTensor(),
                                               T.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])
                                              ])

        # Set mask transforms
        self.mask_transforms = mask_transforms
        if self.mask_transforms is None:
            self.mask_transforms = T.Compose([T.Resize(224),
                                              T.CenterCrop(224),
                                              T.ToTensor()
                                             ])

        # Load image paths
        self.image_directory_path = image_directory_path
        self.image_paths = []
        for file in os.listdir(self.image_directory_path):
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
            mask = self.mask_transforms(image)
            image_classification = 0
        else:
            mask = torch.zeros([1, image.shape[1], image.shape[2]])
            image_classification = 1
            
        return image, image_classification, mask 
