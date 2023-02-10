# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

from cython.parallel import prange
import numpy as np
import time

cdef int mandelbrot(int x, int y, int max_iter) nogil:
    cdef double real, imag
    cdef double real2, imag2
    cdef int i
    real = 1.5 * (x - 500) / (0.5 * 1000)
    imag = (y - 500) / (0.5 * 1000)
    real2, imag2 = real, imag
    for i in range(max_iter):
        real2, imag2 = real2 * real2 - imag2 * imag2 + real, 2 * real2 * imag2 + imag
        if real2 * real2 + imag2 * imag2 > 4:
            return i
    return max_iter

def create_fractal_cython_threaded(int size, int max_iter):
    """
    Create a fractal image using the mandelbrot algorithm
    using a threaded cython version (with openmp)
    :param size: size of the image
    :param max_iter: maximum number of iterations
    """
    cdef int[:, ::1] image = np.zeros((size, size), dtype=np.int32)
    cdef int i, j
    with nogil:
        for i in prange(size):
            for j in range(size):
                image[i, j] = mandelbrot(j, i, max_iter)
    return image

def create_fractal_cython_nonthreaded(int size, int max_iter):
    """
    Create a fractal image using the mandelbrot algorithm
    using a nonthreaded cython version
    :param size: size of the image
    :param max_iter: maximum number of iterations
    """
    cdef int[:, ::1] image = np.zeros((size, size), dtype=np.int32)
    cdef int i, j
    with nogil:
        for i in range(size):
            for j in range(size):
                image[i, j] = mandelbrot(j, i, max_iter)
    return image

def check_acceleration(size: int = 128, max_iter: int = 10):
    """
    Check the acceleration of the Cython code threaded vs non-threaded
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :return: tuple of the time taken for the threaded and non-threaded version
    """
    start = time.time()
    create_fractal_cython_threaded(size, max_iter)
    end = time.time()
    delta_threaded = end - start
    print(f"Cython threaded took: {round(delta_threaded*1000., 3)}ms")
    start = time.time()
    create_fractal_cython_nonthreaded(size, max_iter)
    end = time.time()
    delta_nonthreaded = end - start
    print(f"Cython nonthreaded took {round(delta_nonthreaded*1000., 3)}ms")
    print(f"Threaded is {round(delta_threaded/delta_nonthreaded, 2)}x faster")
    return delta_threaded, delta_nonthreaded
