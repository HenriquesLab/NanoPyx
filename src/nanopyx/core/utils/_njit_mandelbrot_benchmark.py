import warnings

import numpy as np

try:
    from numba import njit, prange

except ImportError:
    # raise a warning that numba is not installed
    # and that the njit functions will not be used
    # and that the pure python functions will be used instead
    warnings.warn("Numba is not installed. Using pure python functions instead.")

    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


def mandelbrot(r, c, max_iter, divergence):
    zr = 0.0
    zi = 0.0
    zr2 = 0.0
    zi2 = 0.0
    i = 0
    while i < max_iter and zr2 + zi2 < divergence:
        zi = 2.0 * zr * zi + c / 500 - 1
        zr = zr2 - zi2 + r / 500 - 1.5
        zr2 = zr * zr
        zi2 = zi * zi
        i += 1
    return i


@njit(cache=True)
def _njit_mandelbrot(r, c, max_iter, divergence):
    zr = 0.0
    zi = 0.0
    zr2 = 0.0
    zi2 = 0.0
    i = 0
    while i < max_iter and zr2 + zi2 < divergence:
        zi = 2.0 * zr * zi + c / 500 - 1
        zr = zr2 - zi2 + r / 500 - 1.5
        zr2 = zr * zr
        zi2 = zi * zi
        i += 1
    return i


def create_fractal_python(size: int, max_iter: int, divergence: float = 4):
    """
    Create a fractal image using the mandelbrot algorithm
    using pure python
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :param divergence: divergence threshold
    """
    image = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            image[i, j] = mandelbrot(j, i, max_iter, divergence)
    return image


@njit(cache=True)
def create_fractal_njit_nonthreaded(size: int, max_iter: int, divergence: float = 4):
    """
    Create a fractal image using the mandelbrot algorithm
    using numba jit
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :param divergence: divergence threshold
    """
    image = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            image[i, j] = _njit_mandelbrot(j, i, max_iter, divergence)
    return image


@njit(cache=True, parallel=True)
def create_fractal_njit_threaded(size: int, max_iter: int, divergence: float = 4):
    """
    Create a fractal image using the mandelbrot algorithm
    using numba jit and parallel
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :param divergence: divergence threshold
    """
    image = np.zeros((size, size), dtype=np.int32)
    for i in prange(size):
        for j in range(size):
            image[i, j] = _njit_mandelbrot(j, i, max_iter, divergence)
    return image
