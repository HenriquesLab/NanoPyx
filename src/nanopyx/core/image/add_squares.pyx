# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=False

from cython.parallel import prange

from ..utils.random cimport _random

import numpy as np
cimport numpy as np

def add_squares(float[:,:] image, float vmax=100, float vmin=0, int nSquares=100):
    """
    Add random squares to an image
    :param image: The image to add the squares to
    :param vmax: The maximum intensity value of the squares
    :param vmin: The minimum intensity value of the squares
    :param nSquares: The number of squares to add
    """
    
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef int n, i, j, x0, x1, y0, y1
    cdef float v

    cdef int[:] x0_arr = np.random.randint(low=0, high=w-1, size=nSquares, dtype=np.int32)
    cdef int[:] x1_arr = np.random.randint(low=0, high=w-1, size=nSquares, dtype=np.int32)
    cdef int[:] y0_arr = np.random.randint(low=0, high=h-1, size=nSquares, dtype=np.int32)
    cdef int[:] y1_arr = np.random.randint(low=0, high=h-1, size=nSquares, dtype=np.int32)
    
    with nogil:
        for n in prange(nSquares):
            v = _random() * (vmax-vmin) + vmin
            x0 = min(x0_arr[n], x1_arr[n])
            x1 = max(x0_arr[n], x1_arr[n])
            y0 = min(y0_arr[n], y1_arr[n])
            y1 = max(y0_arr[n], y1_arr[n])    
            for j in range(y0, y1):
                for i in range(x0, x1):
                    image[j, i] += v
    
    return image


def get_squares(int w, int h, float vmax=100, float vmin=0, int nSquares=100):
    """
    Return an image with random squares
    :param w: The width of the image
    :param h: The height of the image
    :param vmax: The maximum intensity value of the squares
    :param vmin: The minimum intensity value of the squares
    :param nSquares: The number of squares to add
    :return: The image with random squares

    >>> im_squares = get_squares(100, 100)
    >>> im_squares.shape
    (100, 100)
    """

    image = np.zeros((w, h), dtype='float32')
    add_squares(image, vmax, vmin, nSquares)
    return image