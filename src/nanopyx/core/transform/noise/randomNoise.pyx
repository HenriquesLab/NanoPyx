# cython: infer_types=True
# code based on NanoJ-Core/Core/src/nanoj/core2/NanoJRandomNoise.java

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport pow, log, sqrt, exp, pi, floor, fabs, fmax, fmin
import cython as cy
    
@cy.cdivision(True)
cdef double random() nogil:
    # not thread safe since it depends on a time seed
    return rand() / RAND_MAX

@cy.cdivision(True)
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

@cy.cdivision(True)
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

@cy.cdivision(True)
cdef double normalValue() nogil:
    cdef double u = random() * 2 - 1
    cdef double v = random() * 2 - 1
    cdef double r = u * u + v * v
    if r == 0 or r > 1:
        return normalValue()
    cdef double c = sqrt(-2 * log(r) / r)
    return u * c

@cy.boundscheck(False)
@cy.wraparound(False)
def addMixedGaussianPoissonNoise(float[:] image, double gaussSigma, double gaussMean):
    # consider using with arr.ravel() to get a flattened view of the array
    cdef float v
    cdef int l = image.shape[0]
    
    for i in range(l):
        v = image[i]
        v = poissonValue(v) + normalValue() * gaussSigma + gaussMean
        v = fmax(v, 0)
        v = fmin(v, 65535)
        image[i] = v
