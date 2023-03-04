# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

import numpy as np
cimport numpy as np

from cython.parallel import prange

from .interpolation_bilinear cimport _interpolate as _interpolate_bilinear
from .interpolation_nearest_neighbor cimport Interpolator as InterpolatorNearestNeighbor


# bicubic spline interpolation of a 2D array
cdef double _interpolate(float[:,:] image, double x, double y) nogil:
    """
    Bicubic spline interpolation of a 2D array
    :param image: image to interpolate
    :param x: x-coordinate to interpolate at
    :param y: y-coordinate to interpolate at
    :return: interpolated pixel value
    """

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]

    # return 0 if x OR y positions do not exist in image
    if not 0 <= x < w or not 0 <= y < h:
        return 0

    cdef int x0 = int(x)
    cdef int y0 = int(y)

    # do not interpolate if x and y positions exist in image
    if x == x0 and y == y0:
        return image[y0, x0]

    #if x0 < 1 or x0 > w - 3 or y0 < 1 or y0 > h - 3:
    #    return _interpolate_bilinear(image, x, y)

    cdef:
        int i, j, _x, _y
        double[4] a, b
        double dx = x - x0
        double dy = y - y0
        double dx2 = dx * dx
        double dy2 = dy * dy
        double dx3 = dx2 * dx
        double dy3 = dy2 * dy

    a[0] = -0.5 * dx3 + dx2 - 0.5 * dx
    a[1] = 1.5 * dx3 - 2.5 * dx2 + 1
    a[2] = -1.5 * dx3 + 2 * dx2 + 0.5 * dx
    a[3] = 0.5 * dx3 - 0.5 * dx2
    b[0] = -0.5 * dy3 + dy2 - 0.5 * dy
    b[1] = 1.5 * dy3 - 2.5 * dy2 + 1
    b[2] = -1.5 * dy3 + 2 * dy2 + 0.5 * dy
    b[3] = 0.5 * dy3 - 0.5 * dy2

    cdef double result = 0

    for i in range(4):
        for j in range(4):
            # _x = (x0-1+i) % w
            # _y = (y0-1+j) % h
            _x = max(0, min((x0-1+i)%w, w-1))
            _y = max(0, min((y0-1+j)%h, h-1))
            result += a[i] * b[j] * image[_y, _x]

    return result

cdef class Interpolator(InterpolatorNearestNeighbor):

    cdef float _interpolate(self, float x, float y) nogil:
        return _interpolate(self.image, x, y)
