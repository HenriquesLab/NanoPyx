# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

from .nearest_neighbor cimport Interpolator as InterpolatorNearestNeighbor


def interpolate(np.ndarray im, double x, double y) -> float:
    """
    Interpolate image using Catmull-Rom interpolation
    :param im: image to interpolate
    :param x: x-coordinate to interpolate at
    :param y: y-coordinate to interpolate at
    :return: Interpolated pixel value (float)
    """
    return _interpolate(im.view(np.float32), x, y).astype(im.dtype)


cdef double _interpolate(float[:,:] image, double x, double y) nogil:
    """
    Interpolate image using Catmull-Rom interpolation
    :param image: image to interpolate
    :param x: x-coordinate to interpolate at
    :param y: y-coordinate to interpolate at
    :return: Interpolated pixel value (float)
    """
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]

    # return 0 if x and y positions do not exist in image
    if not 0 <= x < w and not 0 <= y < h:
        return 0

    cdef int x0 = int(x)
    cdef int y0 = int(y)    

    # do not interpolate if x and y positions exist in image
    #if x == x0 and y == y0:
    #    return image[y0, x0]

    cdef int u0 = int(x - 0.5)
    cdef int v0 = int(y - 0.5)
    cdef double q = 0
    cdef double p
    cdef int v, u, i, j, _u, _v

    for j in range(4):
        v = v0 - 1 + j
        p = 0
        for i in range(4):
            u = u0 - 1 + i
            _u = max(0, min(u, w-1))
            _v = max(0, min(v, h-1))
            p = p + image[_v, _u] * _cubic(x - (u + 0.5))
        q = q + p * _cubic(y - (v + 0.5))

    #if isnan(q) or isinf(q):
    #    return 0.

    return float(q)


cdef double _cubic(double x) nogil:
    """
    Cubic function used in Catmull-Rom interpolation.
    :param x: the input value
    :return: the output value of cubic function (double)
    """
    cdef double a = 0.5  # Catmull-Rom interpolation
    cdef double z = 0
    if x < 0:
        x = -x
    if x < 1:
        z = x * x * (x * (-a + 2.) + (a - 3.)) + 1.
    elif x < 2:
        z = -a * x * x * x + 5. * a * x * x - 8. * a * x + 4.0 * a
    return z


cdef class Interpolator(InterpolatorNearestNeighbor):

    cdef float _interpolate(self, float x, float y) nogil:
        return _interpolate(self.image, x, y)