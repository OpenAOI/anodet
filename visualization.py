import cv2
import numpy as np
from skimage.segmentation import mark_boundaries





def getBatchVisImage(images, score_maps, thresh, scores):

    indent = 20
    padding = 30
    
    max_height = 0
    tot_width = 0
    for image in images:
        if image.shape[0] > max_height:
            max_height = image.shape[0]
        tot_width += image.shape[1]
    
    

    
    image_tot = np.ones((max_height+3*indent, tot_width + padding*(len(images))+ 2*indent*(len(images)+1), 3))*255
    images = images[::-1]
    score_maps = score_maps[::-1]
    scores = scores[::-1]
    
    last_x = 0
    for i in range(len(images)):

        frame = np.zeros((images[i].shape[1]+2*indent, images[i].shape[0]+2*indent, 3))
        if scores[i]==0:
            frame[:,:,0] = np.ones((images[i].shape[1]+2*indent, images[i].shape[0]+2*indent))*255
        else:
            frame[:,:,1] = np.ones((images[i].shape[1]+2*indent, images[i].shape[0]+2*indent))*255
        

        mask = score_maps[i].copy()
        mask[mask > thresh] = 255
        mask[mask <= thresh] = 0

#         resized = cv2.resize(images[i], (56,56), interpolation = cv2.INTER_AREA)
        image = images[i].copy()


        transparent = np.zeros(mask.shape)
        line_img = mark_boundaries(transparent, mask, color=(1, 0, 0), mode='thick')
        line_img = cv2.resize(line_img, (image.shape[1],image.shape[0]), interpolation = cv2.INTER_AREA)
        line_img = (line_img*255).astype(np.uint8)
        

        a = line_img == [255,0,0]
        a = a[:,:,0]
    

        image[:,:,0] = np.where(a, line_img[:,:,0], image[:,:,0])
        image[:,:,1] = np.where(a, 0, image[:,:,1])
        image[:,:,2] = np.where(a, 0, image[:,:,1])
        
        
#         vis_img = (vis_img*255).astype(np.uint8)


        frame[indent:indent+image.shape[0],indent:indent+image.shape[1]] = image
        
        image_tot[0:frame.shape[0], last_x:last_x + frame.shape[1],:] = frame
        last_x = last_x + frame.shape[1]+padding
        image_tot = image_tot.astype(np.uint8)
        

    return image_tot