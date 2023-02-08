# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from libc.math cimport pi, fabs, sin, floor, ceil, M_PI

import numpy as np
cimport numpy as np

from cython.parallel import prange

from .nearest_neighbor cimport Interpolator as InterpolatorNearestNeighbor

cdef double _interpolate(float[:,:] image, double x, double y, int taps) nogil:
    """
    Interpolate the value of a 2D image at the given coordinates using the Lanczos interpolation method
    :param image: The 2D image to interpolate
    :param x: The x coordinate of the point to interpolate
    :param y: The y coordinate of the point to interpolate
    :param taps: The number of taps (interpolation points) to use in the Lanczos kernel
    :return: The interpolated value at the given coordinates
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

    cdef double x_factor, y_factor
    cdef int i, j
    
    # Determine the low and high indices for the x and y dimensions
    cdef int x_low = int(floor(x) - taps)
    cdef int x_high = int(ceil(x) + taps)
    cdef int y_low = int(floor(y) - taps)
    cdef int y_high = int(ceil(y) + taps)
    
    # Initialize the interpolation value to 0
    cdef double interpolation = 0
    cdef double weight
    cdef double weight_sum = 0
    
    # Loop over the taps in the x and y dimensions
    for i in range(x_low, x_high+1):
        x_factor = _lanczos_kernel(x - i, taps)
        for j in range(y_low, y_high+1):
            y_factor = _lanczos_kernel(y - j, taps)                        
            # Check if the indices are in bounds
            i = max(0, min(i, w-1))
            j = max(0, min(j, h-1))

            # Add the contribution from this tap to the interpolation
            weight = x_factor * y_factor
            interpolation += image[j, i] * weight
            weight_sum += weight
    
    return float(interpolation / weight_sum)


# Lanczos kernel function
cdef double _lanczos_kernel(double x, int taps) nogil:
    """
    Calculate the Lanczos kernel (windowed sinc function) value for a given value.
    REF: https://en.wikipedia.org/wiki/Lanczos_resampling
    :param x: The value for which to calculate the kernel.
    :param taps: The number of taps (interpolation points) in the kernel.
    :return: The kernel value for the given value.
    """
    if x == 0:
        return 1.0
    elif fabs(x) < taps:
        return taps * sin(pi * x) * sin(M_PI * x / taps) / (M_PI * M_PI * x * x)
    else:
        return 0.0


cdef class Interpolator(InterpolatorNearestNeighbor):
    # autogen_pxd: cdef int taps

    def __init__(self, image, int taps=3):
        """
        Interpolate the value of a 2D image at the given coordinates using the Lanczos interpolation method
        :param image: The 2D image to interpolate
        :param taps: The number of taps (interpolation points) to use in the Lanczos kernel
        """
        super().__init__(image)
        self.taps = taps

    cdef float _interpolate(self, float x, float y) nogil:
        """
        Interpolate the value of a 2D image at the given coordinates using the Lanczos interpolation method
        :param x: The x coordinate of the point to interpolate
        :param y: The y coordinate of the point to interpolate
        :return: The interpolated value at the given coordinates
        """
        return _interpolate(self.image, x, y, self.taps)
