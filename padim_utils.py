import torch
from torchvision import transforms as T
from PIL import Image


def toBatch(images, device):
    resize = 256
    cropsize = 224
    transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                  T.CenterCrop(cropsize),
                                  T.ToTensor(),
                                  T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                            ])

    batch = None
    
    for image in images:
        x = Image.fromarray(image).convert('RGB')
        x = transform_x(x).unsqueeze(0)
        
        if batch == None:
            batch = x
        else:
            batch = torch.cat((batch, x), 0)
               
    batch = batch.to(device)
    return batch