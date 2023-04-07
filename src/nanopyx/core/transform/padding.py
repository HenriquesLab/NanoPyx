import numpy as np

def pad_w_zeros_2d(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Pads a 2D NumPy array with zeros to a target height and width.

    :param img: input 2D array
    :type img: numpy.ndarray
    :param height: target height of padded array
    :type height: int
    :param width: target width of padded array
    :type width: int
    :return: padded array
    :rtype: numpy.ndarray
    """
    padded_img = np.zeros((height, width), dtype=img.dtype, mode=img.flags['C_CONTIGUOUS'])
    img_h, img_w = img.shape
    padded_img[(height-img_h)//2:(height-img_h)//2+img_h,(width-img_w)//2:(width-img_w)//2+img_w] = img
    
    return padded_img
