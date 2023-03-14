# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from .omp cimport omp_get_num_procs

import time

import numpy as np
from cython.parallel import prange


cdef int _mandelbrot(double row, double col, int max_iter, double divergence) nogil:
    cdef double real, imag
    cdef double real2, imag2
    cdef int i
    real = 1.5 * (row - 500) / (0.5 * 1000)
    imag = (col - 500) / (0.5 * 1000)
    real2, imag2 = real, imag
    for i in range(max_iter):
        real2, imag2 = real2 * real2 - imag2 * imag2 + real, 2 * real2 * imag2 + imag
        if real2 * real2 + imag2 * imag2 > divergence:
            return i
    return max_iter


def create_fractal_cython_threaded(int size, int max_iter, divergence: float=4):
    """
    Create a fractal image using the mandelbrot algorithm
    using a threaded cython version (with openmp)
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :param divergence: divergence threshold
    """
    cdef int[:, ::1] image = np.zeros((size, size), dtype=np.int32)
    cdef int i, j
    cdef float _divergence = divergence

    print("Number of OMP processors: ", omp_get_num_procs())

    with nogil:
        for i in prange(size):
            for j in range(size):
                image[i, j] = _mandelbrot(j, i, max_iter, _divergence)
    return image


def create_fractal_cython_nonthreaded(int size, int max_iter, divergence: float=4):
    """
    Create a fractal image using the mandelbrot algorithm
    using a nonthreaded cython version
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :divergence: divergence threshold
    """
    cdef int[:, ::1] image = np.zeros((size, size), dtype=np.int32)
    cdef int i, j
    cdef float _divergence = divergence
    with nogil:
        for i in range(size):
            for j in range(size):
                image[i, j] = _mandelbrot(j, i, max_iter, _divergence)
    return image


def create_fractal_opencl(int size, int max_iter, divergence: float=4):
    """
    Create a fractal image using the mandelbrot algorithm
    using opencl
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :param divergence: divergence threshold
    """
    from ...opencl._cl_mandelbrot_benchmark import mandelbrot
    return mandelbrot(size, max_iter, divergence)


def check_acceleration(size: int = 1000, max_iter: int = 1000, divergence: float = 4):
    """
    Check the acceleration of the Cython code threaded vs non-threaded
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :return: tuple of the time taken for the threaded and non-threaded version
    """
    start = time.time()
    create_fractal_cython_threaded(size, max_iter, divergence)
    end = time.time()
    delta_threaded = end - start
    print(f"Cython threaded took: {round(delta_threaded*1000., 3)}ms")
    start = time.time()
    create_fractal_cython_nonthreaded(size, max_iter, divergence)
    end = time.time()
    delta_nonthreaded = end - start
    print(f"Cython nonthreaded took {round(delta_nonthreaded*1000., 3)}ms")
    start = time.time()
    create_fractal_opencl(size, max_iter, divergence)
    end = time.time()
    delta_cl = end - start
    print(f"OpenCL took {round(delta_cl*1000., 3)}ms")

    print(f"Cython-Threaded is {round(delta_nonthreaded/delta_threaded, 2)}x faster than Cython-Non-Threaded")
    print(f"OpenCL is {round(delta_nonthreaded/delta_cl, 2)}x faster than Cython-Non-Threaded")
    print(f"OpenCL is {round(delta_threaded/delta_cl, 2)}x faster than Cython-Threaded")
    return delta_threaded, delta_nonthreaded
