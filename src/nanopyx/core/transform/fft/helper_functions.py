import numpy as np


def check_even_square(image_arr):
    width = image_arr.shape[2]
    height = image_arr.shape[1]

    if width != height:
        return False
    
    if width % 2 != 0:
        return False

    return True

def get_closest_even_square_size(image_slice):
    width = image_slice.shape[1]
    height = image_slice.shape[0]
    min_size = min(width, height)
    
    if min_size % 2 != 0:
        min_size -= 1
    
    return min_size

def make_even_square(image_slice):
    if check_even_square(np.array([image_slice])):
        return image_slice

    width = image_slice.shape[1]
    height = image_slice.shape[0]
    min_size = get_closest_even_square_size(image_slice)

    height_start = int((height-min_size)/2)
    if (height-min_size)%2 != 0:
        height_finish = height - int((height-min_size)/2) - 1
    else:
        height_finish = height - int((height-min_size)/2)

    width_start = int((width-min_size)/2)
    if (width-min_size)%2 != 0:
        width_finish = width - int((width-min_size)/2) -1
    else:
        width_finish = width - int((width-min_size)/2)

    return image_slice[height_start:height_finish, width_start:width_finish]

def flip(ccm):
    dims = ccm.shape
    w1 = dims[1] - 1
    h1 = dims[0] - 1
    pixels = ccm.reshape((dims[0] * dims[1]))
    pixels_out = np.zeros((dims[0] * dims[1]))

    for p0 in range(dims[0] * dims[1]):
        x0 = int(p0 % dims[1])
        y0 = int(p0 / dims[1])
        x1 = w1 - x0
        y1 = h1 - y0
        p1 = int(y1 * dims[0] + x1)
        pixels_out[p0] = pixels[p1]

    return pixels_out.reshape((dims[0], dims[1]))

