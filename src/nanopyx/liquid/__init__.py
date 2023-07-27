"""
The liquid module contains the core functionality of the liquid engine.

The liquid engine is a NanoPyx library specialized in adaptive processing, where the member classes are able to
benchmark themselves and compare their performance for several implementations of the same functionality (OpenCL,
Cython, Numba, etc.).

>>> from nanopyx.liquid import MandelbrotBenchmark
>>> mb = MandelbrotBenchmark()
>>> values = mb.benchmark(64) # doctest: +SKIP
"""

# flake8: noqa: F401

import os
import warnings
import platform

from .__njit__ import njit_works
from .__opencl__ import cl, cl_array, opencl_works, print_opencl_info, devices
from ._le_interpolation_bicubic import ShiftAndMagnify as BCShiftAndMagnify
from ._le_interpolation_bicubic import ShiftScaleRotate as BCShiftScaleRotate
from ._le_interpolation_catmull_rom import ShiftAndMagnify as CRShiftAndMagnify
from ._le_interpolation_catmull_rom import ShiftScaleRotate as CRShiftScaleRotate
from ._le_interpolation_lanczos import ShiftAndMagnify as LZShiftAndMagnify
from ._le_interpolation_lanczos import ShiftScaleRotate as LZShiftScaleRotate
from ._le_interpolation_nearest_neighbor import ShiftAndMagnify as NNShiftAndMagnify
from ._le_interpolation_nearest_neighbor import ShiftScaleRotate as NNShiftScaleRotate
from ._le_mandelbrot_benchmark import MandelbrotBenchmark
from ._le_radiality import Radiality
from ._le_radial_gradient_convergence import RadialGradientConvergence
from ._le_roberts_cross_gradients import GradientRobertsCross
from ._le_esrrf import eSRRF as eSRRF_ST
from ._le_convolution import Convolution as Convolution2D
from ._le_DUMMY import DUMMY

from multiprocessing import current_process

if current_process().name == 'MainProcess':
        
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{__name__}.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # OS INFO
    os_msg = "\n" + "=" * 60 + "\nOS INFORMATION\n"
    os_msg += "=" * 60 + "\n"
    os_msg += "OS:  " + platform.platform()  + "\n"
    os_msg += "Architecture: " + platform.machine() + "\n"
        
    # CPU INFO
    # TODO this can be done with shell commands on a platform dependent manner or external libraries
    cpu_msg = "\n" + "=" * 60 + "\nCPU INFORMATION\n"
    cpu_msg += "=" * 60 + "\n"
    cpu_msg += "CPU:  " + platform.processor()  + "\n"


    # RAM INFO
    # TODO this can be done with shell commands on a platform dependent manner or external libraries
    ram_msg = "\n" + "=" * 60 + "\nRAM INFORMATION\n"
    ram_msg += "=" * 60 + "\n"
    ram_msg += "RAM:  " + "TBD"  + "\n"

    # GPU INFO
    if opencl_works():
        gpu_msg = print_opencl_info()
    else:
        gpu_msg = "\n" + "=" * 60 + "\n NO PYOPENCL SUPPORT\n"
        gpu_msg += "=" * 60 + "\n"

    # PYTHON INFO
    py_msg = "\n" + "=" * 60 + "\nPYTHON INFORMATION\n"
    py_msg += "=" * 60 + "\n"
    py_msg += "Version:  " + platform.python_version()  + "\n"
    py_msg += "Implementation:  " + platform.python_implementation()  + "\n"
    py_msg += "Compiler:  " + platform.python_compiler()  + "\n"


    logger.info(os_msg)
    logger.info(cpu_msg)
    logger.info(ram_msg)
    logger.info(gpu_msg)
    logger.info(py_msg)