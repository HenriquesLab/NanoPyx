# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

from cython.parallel import prange

from .nearest_neighbor cimport _interpolate as _interpolate_nn
from .nearest_neighbor cimport Interpolator as InterpolatorNearestNeighbor

# bilinear spline interpolation of a 2D array
cdef double _interpolate(float[:,:] image, double x, double y) nogil:
    """
    Bilinear spline interpolation of a 2D array
    :param image: The image to be interpolated
    :param x, y: The coordinates at which to interpolate the image
    :return: The interpolated value of the image at the given coordinates
    """

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]

    # return 0 if x and y positions do not exist in image
    if not 0 <= x < w and not 0 <= y < h:
        return 0

    cdef int x0 = int(x)
    cdef int y0 = int(y)    

    # do not interpolate if x and y positions exist in image
    if x == x0 and y == y0:
        return image[y0, x0]

    cdef:
        int x1 = min(max(0, x0 + 1), w - 2)
        int y1 = min(max(0, y0 + 1), h - 2)
        float v0 = image[y0, x0]
        float v1 = image[y0, x1]
        float v2 = image[y1, x0]
        float v3 = image[y1, x1]
        double dx = x - x0
        double dy = y - y0
        double dx1 = 1.0 - dx
        double dy1 = 1.0 - dy
        double v01 = v0 * dx1 + v1 * dx
        double v23 = v2 * dx1 + v3 * dx
    
    return v01 * dy1 + v23 * dy


cdef class Interpolator(InterpolatorNearestNeighbor):

    cdef float _interpolate(self, float x, float y) nogil:
        return _interpolate(self.image, x, y)
