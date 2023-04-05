# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

import numpy as np
cimport numpy as np

cdef float _interpolate(float[:,:] image, float x, float y) nogil:
    """
    Interpolate image using Lanczos interpolation
    :param image: image to interpolate
    :param x: x-coordinate to interpolate at
    :param y: y-coordinate to interpolate at
    :return: Interpolated pixel value (float)
    """
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]

    return _c_interpolate(&image[0,0], y, x, rows, cols)


cdef class Interpolator(InterpolatorNearestNeighbor):

    cdef float _interpolate(self, float x, float y) nogil:
        return _interpolate(self.image, x, y)

