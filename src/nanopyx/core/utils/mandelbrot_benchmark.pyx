# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from .omp cimport omp_get_num_procs

import time

import numpy as np
from cython.parallel import prange

from ...liquid import MandelbrotBenchmark
from ...opencl import works as opencl_works
from ...opencl._cl_mandelbrot_benchmark import _cl_mandelbrot
from ._njit_mandelbrot_benchmark import (create_fractal_njit_nonthreaded,
                                         create_fractal_njit_threaded,
                                         create_fractal_python)


def check_acceleration(size: int = 1000):
    """
    Check the acceleration of the opencl vs cython version
    :param size: size of the image
    :return: tuple of images
    """
    bench = MandelbrotBenchmark()
    images = bench.benchmark(size=size)
    return images
