# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from .omp cimport omp_get_num_procs

import time

import numpy as np
from cython.parallel import prange

from ...opencl import works as opencl_works
from ._njit_mandelbrot_benchmark import (create_fractal_njit_nonthreaded,
                                         create_fractal_njit_threaded,
                                         create_fractal_python)


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
                image[i, j] = _c_mandelbrot(j, i, max_iter, _divergence)
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
                image[i, j] = _c_mandelbrot(j, i, max_iter, _divergence)
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
    Check the acceleration of the cython version compared to the python version and the opencl version
    :param size: size of the image
    :param max_iter: maximum number of iterations
    :return: tuple of image and speed of the cython version compared to the python version and the opencl version
    """
    msg_timing = "Benchmarking:\n"
    start = time.time()
    im_python = create_fractal_python(size, max_iter, divergence)
    end = time.time()
    delta_python= end - start
    msg_timing += f"- Python took: {round(delta_python*1000., 3)}ms\n"
    start = time.time()
    im_njit_nonthreaded = create_fractal_njit_nonthreaded(size, max_iter, divergence)
    end = time.time()
    delta_njit_nonthreaded= end - start
    msg_timing += f"- NJIT-Nonthreaded - took: {round(delta_python*1000., 3)}ms\n"
    start = time.time()
    im_njit_threaded = create_fractal_njit_threaded(size, max_iter, divergence)
    end = time.time()
    delta_njit_threaded= end - start
    msg_timing += f"- NJIT-Threaded - took: {round(delta_python*1000., 3)}ms\n"
    start = time.time()
    im_cython_nonthreaded = create_fractal_cython_nonthreaded(size, max_iter, divergence)
    end = time.time()
    delta_cython_nonthreaded = end - start
    msg_timing += f"- Cython nonthreaded took: {round(delta_cython_nonthreaded*1000., 3)}ms\n"
    start = time.time()
    im_cython_threaded = create_fractal_cython_threaded(size, max_iter, divergence)
    end = time.time()
    delta_cython_threaded = end - start
    msg_timing += f"- Cython threaded took: {round(delta_cython_threaded*1000., 3)}ms\n"

    msg_comparison = "Comparison:\n"
    msg_comparison += f"- NJIT-Nonthreaded is {round(delta_python/delta_njit_nonthreaded, 2)}x faster than Pure Python\n"
    msg_comparison += f"- NJIT-Threaded is {round(delta_python/delta_njit_threaded, 2)}x faster than Pure Python\n"
    msg_comparison += f"- Cython-Nonthreaded is {round(delta_python/delta_cython_nonthreaded, 2)}x faster than Pure Python\n"
    msg_comparison += f"- Cython-Nonthreaded is {round(delta_njit_nonthreaded/delta_cython_nonthreaded, 2)}x faster than NJIT-Nonthreaded\n"
    msg_comparison += f"- Cython-Nonthreaded is {round(delta_njit_threaded/delta_cython_nonthreaded, 2)}x faster than NJIT-Threaded\n"
    msg_comparison +=  f"- Cython-Threaded is {round(delta_python/delta_cython_threaded, 2)}x faster than Pure Python\n"
    msg_comparison +=  f"- Cython-Threaded is {round(delta_njit_nonthreaded/delta_cython_threaded, 2)}x faster than NJIT-Nonthreaded\n"
    msg_comparison +=  f"- Cython-Threaded is {round(delta_njit_threaded/delta_cython_threaded, 2)}x faster than NJIT-Threaded\n"
    msg_comparison +=  f"- Cython-Threaded is {round(delta_cython_nonthreaded/delta_cython_threaded, 2)}x faster than Cython-Nonthreaded\n"

    # check if opencl works
    delta_cl = 0
    im_cl = np.zeros_like(im_python)
    if opencl_works():
        start = time.time()
        im_cl = create_fractal_opencl(size, max_iter, divergence)
        end = time.time()
        delta_cl = end - start
        msg_timing += f"- OpenCL took: {round(delta_cl*1000., 3)}ms\n"
        msg_comparison += f"- OpenCL is {round(delta_python/delta_cl, 2)}x faster than Pure Python\n"
        msg_comparison += f"- OpenCL is {round(delta_njit_nonthreaded/delta_cl, 2)}x faster than NJIT-Nonthreaded\n"
        msg_comparison += f"- OpenCL is {round(delta_njit_threaded/delta_cl, 2)}x faster than NJIT-Threaded\n"
        msg_comparison += f"- OpenCL is {round(delta_cython_nonthreaded/delta_cl, 2)}x faster than Cython-Nonthreaded\n"
        msg_comparison += f"- OpenCL is {round(delta_cython_threaded/delta_cl, 2)}x faster than Cython-Threaded\n"

    print(f"{msg_timing}\n{msg_comparison}")

    return (im_python, delta_python), (im_njit_nonthreaded, delta_njit_threaded), (im_njit_threaded, delta_njit_threaded), (im_cython_nonthreaded, delta_cython_nonthreaded), (im_cython_threaded, delta_cython_threaded), (im_cl, delta_cl)
