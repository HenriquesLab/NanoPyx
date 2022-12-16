# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3
# code based on NanoJ-Core/Core/src/nanoj/core2/NanoJRandomNoise.java

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport pow, log, sqrt, exp, pi, floor, fabs, fmax, fmin

import cython
cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

from noise import pnoise2

r = np.random.RandomState()

# @cython.cdivision(True)
cdef double random() nogil:
    # not thread safe since it depends on a time seed
    return float(rand()) / float(RAND_MAX)

# @cython.cdivision(True)
cdef double logFactorial(int x) nogil:
    """
    Return the logarithm of the factorial of x. Uses Stirling's approximation for large
    values.
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
    
cdef int poissonSmall(double mean) nogil:
    cdef double L = exp(-mean)
    cdef double p = 1.
    cdef int k = 1
    
    p *= random()
    
    while (p > L):
        p *= random()
        k += 1
    
    return k - 1

# @cython.cdivision(True)
cdef int poissonLarge(double mean) nogil:
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
        u = random()
        x = (alpha - log((1.0 - u) / u))/beta
        n = int(floor(x + 0.5))
        if n < 0: 
            continue
        v = random()
        y = alpha - beta*x
        temp = 1.0 + exp(y)
        lhs = y + log(v / (temp * temp))
        rhs = k + n*log(mean) - logFactorial(n)
        if lhs <= rhs:
            return n

cdef int poissonValue(double mean) nogil:
    if mean < 100:
        return poissonSmall(mean)
    else:
        return poissonLarge(mean)

# @cython.cdivision(True)
cdef double normalValue() nogil:
    cdef double u = random() * 2 - 1
    cdef double v = random() * 2 - 1
    cdef double r = u * u + v * v
    if r == 0 or r > 1:
        return normalValue()
    cdef double c = sqrt(-2 * log(r) / r)
    return u * c

# @cython.boundscheck(False)
# @cython.wraparound(False)
def addMixedGaussianPoissonNoise(np.ndarray[np.float32_t, ndim=2] image, double gaussSigma, double gaussMean):
    # consider using with arr.ravel() to get a flattened view of the array

    cdef float v
    cdef int i, j

    with nogil:
        for j in range(image.shape[1]):
            for i in range(image.shape[0]):
                v = image[i,j]
                v = poissonValue(v) + normalValue() * gaussSigma + gaussMean
                v = fmax(v, 0)
                v = fmin(v, 65535)
                image[i,j] = v

def addMixedGaussianPoissonNoise2(np.ndarray image, double gaussSigma, double gaussMean):
    shape = []
    for i in range(image.ndim):
        shape.append(image.shape[i])
    image[:] = np.clip(r.poisson(image)+r.normal(scale=gaussSigma, size=tuple(shape), loc=gaussMean), 0, 65535)

def addPerlinNoise(np.ndarray[np.float32_t, ndim=2] image, int amp=100, float f = 100, int octaves = 1, float persistence = 0.5, float lacunarity = 2., float repeatx = 1024, float repeaty = 1024, int base = 0):
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            image[i, j] += amp * pnoise2(i/f, j/f, octaves, persistence, lacunarity, repeatx, repeaty, base)

def addSquares(np.ndarray[np.float32_t, ndim=2] image, float vmax=100, float vmin=0, int nSquares=100):
    
    cdef int w = image.shape[0]
    cdef int h = image.shape[1]
    cdef int x0, x1, y0, y1
    cdef float v

    cdef int[:] x0_arr = np.random.randint(low=0, high=w-1, size=nSquares, dtype='int32')
    cdef int[:] x1_arr = np.random.randint(low=0, high=w-1, size=nSquares, dtype='int32')
    cdef int[:] y0_arr = np.random.randint(low=0, high=h-1, size=nSquares, dtype='int32')
    cdef int[:] y1_arr = np.random.randint(low=0, high=h-1, size=nSquares, dtype='int32')
    
    with nogil:
        for n in range(nSquares):
            v = random() * (vmax-vmin) + vmin
            x0 = min(x0_arr[n], x1_arr[n])
            x1 = max(x0_arr[n], x1_arr[n])
            y0 = min(y0_arr[n], y1_arr[n])
            y1 = max(y0_arr[n], y1_arr[n])    
            for j in range(y0, y1):
                for i in range(x0, x1):
                    image[i, j] += v
    
    return image

def addRamp(np.ndarray[np.float32_t, ndim=2] image, float vmax=100, float vmin=0):

    cdef int w = image.shape[0]
    cdef int h = image.shape[1]
    cdef float v

    with nogil:
        for j in range(h):
            v = float(j)/h * (vmax-vmin) + vmin
            for i in range(w):
                image[i, j] += v
    