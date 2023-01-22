# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=False

from cython.parallel import prange

import numpy as np
cimport numpy as np


def add_ramp(float[:,:] image, float vmax=100, float vmin=0):
    """
    Adds a ramp from vmin to vmax to the image
    :param image: The image to add the ramp to
    :param vmax: The maximum intensity value of the ramp
    :param vmin: The minimum intensity value of the ramp
    """

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef float v
    cdef int i, j

    with nogil:
        for i in prange(w):
            v = float(i)/w * (vmax-vmin) + vmin
            for j in range(h):
                image[j, i] += v


def get_ramp(int w, int h, float vmax=100, float vmin=0):
    """
    Returns a 2D array of size (w, h) with a ramp from vmin to vmax
    :param w: The width of the image
    :param h: The height of the image
    :param vmax: The maximum intensity value of the ramp
    :param vmin: The minimum intensity value of the ramp
    :return: The image with the ramp
    """
    image = np.zeros((h, w), dtype='float32')
    add_ramp(image, vmax, vmin)
    return image