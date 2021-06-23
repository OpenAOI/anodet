import torch
from torchvision import transforms as T
from PIL import Image


def toBatch(images, device):
#     size = 224
    size = 256
    cropsize = 224
    transform_x = T.Compose([
        T.Resize(cropsize),
        T.CenterCrop(cropsize),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    batch = torch.zeros((len(images), 3, cropsize, cropsize))
    for i in range(len(images)):        
        x = Image.fromarray(images[i]).convert('RGB')
        x = transform_x(x)
        batch[i] = x
               
    batch = batch.to(device)
    return batch