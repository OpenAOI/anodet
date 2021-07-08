import cv2
import numpy as np
from skimage.segmentation import mark_boundaries




def getBoundaryImageClassificationGroup(images, patch_classifications, image_classifications, size):
        
    padding = 30
    margin = 50
    tot_image = np.ones((size+2*padding, len(images)*(size+2*padding)+(len(images)-1)*margin, 3)).astype(np.uint8)*255

    for i in range(len(images)):
        boundary_image = getBoundaryImageClassification(images[i], patch_classifications[i], image_classifications[i], size, padding=padding)
        w = boundary_image.shape[0]
        tot_image[:, i*(w+margin):(i+1)*(w)+i*margin,:] = boundary_image
    
    return tot_image
        
        
        
def getBoundaryImageClassification(image, patch_classification, image_classification, size, padding=20):
    
    frame = (np.ones((size+2*padding, size+2*padding, 3))*255).astype(np.uint8)
    
    if image_classification:
        frame[:,:,0] = 0 
        frame[:,:,2] = 0 
    else:
        frame[:,:,1] = 0 
        frame[:,:,2] = 0 
        
    boundary_image = getBoundaryImage(image, patch_classification, size)
    frame[padding:frame.shape[0]-padding,padding:frame.shape[1]-padding] = boundary_image
    return frame



def getBoundaryImage(image, patch_classification, size):
    
    image = image.copy()
    image = cv2.resize(image, (size,size), interpolation = cv2.INTER_AREA)

    mask = patch_classification.cpu().numpy()
    
    transparent = np.zeros(mask.shape)
    line_img = mark_boundaries(transparent, mask, color=(1, 0, 0), mode='thick')
    line_img = cv2.resize(line_img, (image.shape[1],image.shape[0]), interpolation = cv2.INTER_AREA)
    line_img = (line_img*255).astype(np.uint8)
    
    a = line_img == [255,0,0]
    a = a[:,:,0]
    
    image[:,:,0] = np.where(a, line_img[:,:,0], image[:,:,0])
    image[:,:,1] = np.where(a, 0, image[:,:,1])
    image[:,:,2] = np.where(a, 0, image[:,:,1])

    return image
    