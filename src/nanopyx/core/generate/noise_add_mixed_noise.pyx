# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True
# code based on NanoJ-Core/Core/src/nanoj/core2/NanoJRandomNoise.java

from libc.math cimport pow, log, sqrt, exp, pi, floor, fabs, fmax, fmin

from ..utils.random cimport _random

import cython
cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

r = np.random.RandomState()

cdef double _log_factorial(int x) nogil:
    """
    Calculates the logarithm of the factorial of x. Uses Stirling's approximation for large values.
    :param x: argument
    :return: logarithm of the factorial of x
    """
    cdef double ans = 0
    cdef int i = 0
    cdef double y = x

    if x < 15:
        for i in range(1, x+1):
            ans += log(i)
        return ans
    else:
        ans = y*log(y) + log(2.0*pi*y)/2 - y + (pow(y,-1))/12 - (pow(y,-3))/360 + (pow(y,-5))/1260 - (pow(y,-7))/1680 + (pow(y,-9))/1188
        return ans

cdef int _poisson_small(double mean) nogil:
    """
    Calculates a poisson distributed random value with the specified mean using an algorithm based on the factorial function. This method is more efficient for small means.
    :param mean: mean of the poisson distribution
    :return: random value
    """
    cdef double L = exp(-mean)
    cdef double p = 1.
    cdef int k = 1

    p *= _random()

    while (p > L):
        p *= _random()
        k += 1

    return k - 1


cdef int _poisson_large(double mean) nogil:
    """
    Calculates a poisson distributed random value with the specified mean using the rejection method PA. This method is more efficient for large means.
    :param mean: mean of the poisson distribution
    :return: random value
    """
    # "Rejection method PA" from "The Computer Generation of
    # Poisson Random Variables" by A. C. Atkinson,
    # Journal of the Royal Statistical Society Series C
    # (Applied Statistics) Vol. 28, No. 1. (1979)
    # The article is on pages 29-35.
    # The algorithm given here is on page 32.

    mean = fabs(mean)
    cdef double c = 0.767 - 3.36/mean
    cdef double beta = pi/sqrt(3.0 * mean)
    cdef double alpha = beta*mean
    cdef double k = log(c) - mean - log(beta)

    cdef double u, x, v, y, temp
    cdef int n
    while (True):
        u = _random()
        x = (alpha - log((1.0 - u) / u))/beta
        n = int(floor(x + 0.5))
        if n < 0:
            continue
        v = _random()
        y = alpha - beta*x
        temp = 1.0 + exp(y)
        lhs = y + log(v / (temp * temp))
        rhs = k + n*log(mean) - _log_factorial(n)
        if lhs <= rhs:
            return n

cdef int _poisson_value(double mean) nogil:
    """
    Calculates a poisson distributed random value with the specified mean. Uses a different algorithm for small and large means.
    :param mean: mean of the poisson distribution
    :return: random value
    """
    if mean < 100:
        return _poisson_small(mean)
    else:
        return _poisson_large(mean)


cdef double _normal_value() nogil:
    """
    Returns a normally distributed random value with mean 0 and standard deviation 1.
    """
    cdef double u = _random() * 2 - 1
    cdef double v = _random() * 2 - 1
    cdef double r = u * u + v * v
    if r == 0 or r > 1:
        return _normal_value()
    cdef double c = sqrt(-2 * log(r) / r)
    return u * c


def add_mixed_gaussian_poisson_noise(image, double gauss_sigma, double gauss_mean):
    """
    Add mixed Gaussian-Poisson noise to an image, pure cython version
    :param image: The image to add noise to, need to be 2D or 3D
    :type image: numpy.ndarray or numpy.view
    :param gauss_sigma: The standard deviation of the Gaussian noise
    :type gauss_sigma: float
    :param gauss_mean: The mean of the Gaussian noise
    :type gauss_mean: float
    """

    assert image.ndim == 2 or image.ndim == 3, "Only 2D and 3D images are supported"

    cdef float v
    cdef int i, j, f
    cdef float[:,:] image_2d
    cdef float[:,:,:] image_3d

    if image.ndim == 2:
        image_2d = image
        with nogil:
            for j in prange(image_2d.shape[0]):
                for i in range(image_2d.shape[1]):
                    v = image_2d[j,i]
                    v = _poisson_value(v) + _normal_value() * gauss_sigma + gauss_mean
                    v = fmax(v, 0)
                    v = fmin(v, 65535)
                    image_2d[j,i] = v

    else:
        image_3d = image
        with nogil:
            for f in prange(image_3d.shape[0]):
                for j in range(image_3d.shape[1]):
                    for i in range(image_3d.shape[2]):
                        v = image_3d[f,j,i]
                        v = _poisson_value(v) + _normal_value() * gauss_sigma + gauss_mean
                        v = fmax(v, 0)
                        v = fmin(v, 65535)
                        image_3d[f,j,i] = v


def add_mixed_gaussian_poisson_noise2(np.ndarray image, double gauss_sigma, double gauss_mean):
    """
    Add mixed Gaussian-Poisson noise to an image, pure numpy version
    :param image: The image to add noise to
    :type image: np.ndarray
    :param gauss_sigma: The standard deviation of the Gaussian noise
    :type gauss_sigma: float
    :param gauss_mean: The mean of the Gaussian noise
    :type gauss_mean: float
    """
    shape = []
    for i in range(image.ndim):
        shape.append(image.shape[i])
    image[:] = np.clip(r.poisson(image)+r.normal(scale=gauss_sigma, size=tuple(shape), loc=gauss_mean), 0, 65535)
