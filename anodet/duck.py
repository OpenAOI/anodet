import torchvision
from torchvision import transforms as T
import numpy as np
from PIL import Image
import os

filename = "/Users/helvetica/Desktop/duck_rgb/original.jpg"
with Image.open(filename) as img:
    img.load()

transform = T.ToTensor()

tensor_img = transform(img)

T.Normalize(mean=[1.485, 1.456, 1.406], std=[1.229, 1.224, 1.225])

numpy_array = tensor_img.numpy()

# Flatten to save txt
# flattened_array = numpy_array.flatten()

# save_filename = "/Users/helvetica/Desktop/duck_rgb/tensor_image.txt"
# np.savetxt(save_filename, flattened_array)

save_directory = "/Users/helvetica/Desktop/duck_rgb/arrays/"
os.makedirs(save_directory, exist_ok=True)

for i in range(numpy_array.shape[0]):
    save_filename = os.path.join(save_directory, f"tensor_image_{i}.txt")
    np.savetxt(save_filename, numpy_array[i])