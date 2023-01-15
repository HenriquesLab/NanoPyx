# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True
# nanopyx: autogen_pxd=True

import numpy as np
cimport numpy as np

from cython.parallel import prange

from .nearest_neighbor cimport _interpolate as _interpolate_nn
from .nearest_neighbor cimport Interpolator as InterpolatorNearestNeighbor

# bilinear spline interpolation of a 2D array
cdef double _interpolate(float[:,:] image, double x, double y) nogil:
    """
    Bilinear spline interpolation of a 2D array.

    Parameters
    ----------
    image : 2D array
        The image to be interpolated.
    x, y : float
        The coordinates at which to interpolate the image.

    Returns
    -------
    value : float
        The interpolated value of the image at the given coordinates.
    """

    cdef:
        int x0 = int(x)
        int y0 = int(y)
        int w = image.shape[1]
        int h = image.shape[0]
    
    if x0 < 0 or x0 > w - 2 or y0 < 0 or y0 > h - 2:
        return _interpolate_nn(image, x, y)

    cdef:
        int x1 = x0 + 1
        int y1 = y0 + 1
        double dx = x - x0
        double dy = y - y0
        double dx1 = 1.0 - dx
        double dy1 = 1.0 - dy
        float v0 = image[y0, x0]
        float v1 = image[y0, x1]
        float v2 = image[y1, x0]
        float v3 = image[y1, x1]
        double v01 = v0 * dx1 + v1 * dx
        double v23 = v2 * dx1 + v3 * dx
    
    return v01 * dy1 + v23 * dy


cdef class Interpolator(InterpolatorNearestNeighbor):

    cdef float _interpolate(self, float x, float y) nogil:
        return _interpolate(self.image, x, y)
