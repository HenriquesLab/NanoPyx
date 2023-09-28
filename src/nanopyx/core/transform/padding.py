import numpy as np


def pad_w_zeros_2d(img, height, width):
    """
    Generate a 2D padded image with zeros.

    Parameters:
        img (numpy.ndarray): The input image.
        height (int): The desired height of the padded image.
        width (int): The desired width of the padded image.

    Returns:
        numpy.ndarray: The padded image with zeros.
    """
    padded_img = np.zeros((height, width), dtype=img.dtype)
    img_h, img_w = img.shape
    padded_img[
        (height - img_h) // 2 : (height - img_h) // 2 + img_h, (width - img_w) // 2 : (width - img_w) // 2 + img_w
    ] = img

    return padded_img
