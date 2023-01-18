# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True
# code based on NanoJ-Core/Core/src/nanoj/core2/NanoJRandomNoise.java

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport pow, log, sqrt, exp, pi, floor, fabs, fmax, fmin

import cython
cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

from noise import pnoise2
import opensimplex

r = np.random.RandomState()


cdef double random() nogil:
    """
    Returns a random value between 0 and 1.
    """
    # not thread safe since it depends on a time seed
    return float(rand()) / float(RAND_MAX)


cdef double logFactorial(int x) nogil:
    """
    Return the logarithm of the factorial of x. Uses Stirling's approximation for large values.
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
    """
    Returns a poisson distributed random value with the specified mean using an algorithm based on the factorial function. This method is more efficient for small means.
    """
    cdef double L = exp(-mean)
    cdef double p = 1.
    cdef int k = 1
    
    p *= random()
    
    while (p > L):
        p *= random()
        k += 1
    
    return k - 1


cdef int poissonLarge(double mean) nogil:
    """
    Returns a poisson distributed random value with the specified mean using the rejection method PA. This method is more efficient for large means.
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
    """
    Returns a poisson distributed random value with the specified mean. Uses a different algorithm for small and large means.
    """
    if mean < 100:
        return poissonSmall(mean)
    else:
        return poissonLarge(mean)


cdef double normalValue() nogil:
    """
    Returns a normally distributed random value with mean 0 and standard deviation 1.
    """
    cdef double u = random() * 2 - 1
    cdef double v = random() * 2 - 1
    cdef double r = u * u + v * v
    if r == 0 or r > 1:
        return normalValue()
    cdef double c = sqrt(-2 * log(r) / r)
    return u * c


def addMixedGaussianPoissonNoise(float[:,:] image, double gaussSigma, double gaussMean):
    """
    Add mixed Gaussian-Poisson noise to an image, pure cython version
    :param image: The image to add noise to
    :param gaussSigma: The standard deviation of the Gaussian noise
    :param gaussMean: The mean of the Gaussian noise
    """

    cdef float v
    cdef int i, j

    with nogil:
        for j in prange(image.shape[1]):
            for i in range(image.shape[0]):
                v = image[j,i]
                v = poissonValue(v) + normalValue() * gaussSigma + gaussMean
                v = fmax(v, 0)
                v = fmin(v, 65535)
                image[j,i] = v

def addMixedGaussianPoissonNoise2(np.ndarray image, double gaussSigma, double gaussMean):
    """
    Add mixed Gaussian-Poisson noise to an image, pure numpy version
    :param image: The image to add noise to
    :param gaussSigma: The standard deviation of the Gaussian noise
    :param gaussMean: The mean of the Gaussian noise
    """
    shape = []
    for i in range(image.ndim):
        shape.append(image.shape[i])
    image[:] = np.clip(r.poisson(image)+r.normal(scale=gaussSigma, size=tuple(shape), loc=gaussMean), 0, 65535)


def addPerlinNoise(float[:,:] image, int amp=100, int offset = 100, float f = 100, int octaves = 1, float persistence = 0.5, float lacunarity = 2., float repeatx = 1024, float repeaty = 1024, int base = 0):
    """
    Add perlin noise to an image
    :param image: The image to add noise to
    :param amp: The amplitude of the noise
    :param offset: The offset of the noise
    :param f: The frequency of the noise
    :param octaves: The number of octaves
    :param persistence: The persistence of the noise
    :param lacunarity: The lacunarity of the noise
    :param repeatx: The repeat of the noise in the x direction
    :param repeaty: The repeat of the noise in the y direction
    :param base: The base of the noise
    """
    cdef double p
    cdef int w = image.shape[1]
    cdef int h = image.shape[0]
    cdef float f_x = f / w
    cdef float f_y = f / h
    if w > h:
        f_x *= w / h
    else:
        f_y *= h / w

    for j in range(h):
        for i in range(w):
            p = pnoise2(i * f_x, j * f_y, octaves, persistence, lacunarity, repeatx, repeaty, base)
            image[j, i] += amp * p + offset


def getPerlinNoise(w, h, amp=100, int offset = 100, f = 10, octaves = 1, persistence = 0.5, lacunarity = 2., repeatx = 1024, repeaty = 1024, base = 0):
    """
    Return a perlin noise image
    :param w: The width of the image
    :param h: The height of the image
    :param amp: The amplitude of the noise
    :param offset: The offset of the noise
    :param f: The frequency of the noise
    :param octaves: The number of octaves
    :param persistence: The persistence of the noise
    :param lacunarity: The lacunarity of the noise
    :param repeatx: The repeat of the noise in the x direction
    :param repeaty: The repeat of the noise in the y direction
    :param base: The base of the noise
    :return: The perlin noise image
    """
    image = np.zeros((w, h), dtype=np.float32)
    addPerlinNoise(image, amp, offset, f, octaves, persistence, lacunarity, repeatx, repeaty, base)
    return image


def getSimplexNoise(int w, int h, int f = 1):
    """
    Return a simplex noise image
    REF: https://github.com/lmas/opensimplex
    :param w: The width of the image
    :param h: The height of the image
    :param f: The frequency of the noise
    :return: The simplex noise image
    """

    cdef float f_x = f
    cdef float f_y = f
    if w > h:
        f_y *= w / h
    else:
        f_x *= h / w

    x0 = np.linspace(0, f_x, num = h, dtype=np.float32)
    y0 = np.linspace(0, f_y, num = w, dtype=np.float32)

    return opensimplex.noise2array(x0, y0)


def addSquares(float[:,:] image, float vmax=100, float vmin=0, int nSquares=100):
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
            v = random() * (vmax-vmin) + vmin
            x0 = min(x0_arr[n], x1_arr[n])
            x1 = max(x0_arr[n], x1_arr[n])
            y0 = min(y0_arr[n], y1_arr[n])
            y1 = max(y0_arr[n], y1_arr[n])    
            for j in range(y0, y1):
                for i in range(x0, x1):
                    image[j, i] += v
    
    return image


def getSquares(int w, int h, float vmax=100, float vmin=0, int nSquares=100):
    """
    Return an image with random squares
    :param w: The width of the image
    :param h: The height of the image
    :param vmax: The maximum intensity value of the squares
    :param vmin: The minimum intensity value of the squares
    :param nSquares: The number of squares to add
    :return: The image with random squares
    """

    image = np.zeros((w, h), dtype='float32')
    addSquares(image, vmax, vmin, nSquares)
    return image


def addRamp(float[:,:] image, float vmax=100, float vmin=0):
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


def getRamp(int w, int h, float vmax=100, float vmin=0):
    """
    Returns a 2D array of size (w, h) with a ramp from vmin to vmax
    :param w: The width of the image
    :param h: The height of the image
    :param vmax: The maximum intensity value of the ramp
    :param vmin: The minimum intensity value of the ramp
    :return: The image with the ramp
    """
    image = np.zeros((w, h), dtype='float32')
    addRamp(image, vmax, vmin)
    return image


def test_logFactorial():
    assert abs(logFactorial(0) - 0.0) < 1e-6
    assert abs(logFactorial(1) - 0.0) < 1e-6
    assert abs(logFactorial(2) - 0.6931471805599453) < 1e-6
    assert abs(logFactorial(3) - 1.791759469228055) < 1e-6
    assert abs(logFactorial(4) - 3.1780538303479458) < 1e-6
    assert abs(logFactorial(5) - 4.787491742782046) < 1e-6
    assert abs(logFactorial(6) - 6.579251212010101) < 1e-6




