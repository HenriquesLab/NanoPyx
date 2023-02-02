import numpy as np

def pad_w_zeros_2d(img, height, width):
    
    padded_img = np.zeros((height, width), dtype=img.dtype)
    img_h, img_w = img.shape
    padded_img[(height-img_h)//2:(height-img_h)//2+img_h,(width-img_w)//2:(width-img_w)//2+img_w] = img
    
    return padded_img