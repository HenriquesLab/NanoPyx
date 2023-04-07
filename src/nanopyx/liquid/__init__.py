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

from .__njit__ import njit_works
from .__opencl__ import cl, cl_array, cl_ctx, cl_queue, opencl_works, print_opencl_info
from ._le_interpolation_bicubic import ShiftAndMagnify as BCShiftAndMagnify
from ._le_interpolation_bicubic import ShiftScaleRotate as BCShiftScaleRotate
from ._le_interpolation_catmull_rom import ShiftAndMagnify as CRShiftAndMagnify
from ._le_interpolation_catmull_rom import ShiftScaleRotate as CRShiftScaleRotate
from ._le_interpolation_lanczos import ShiftAndMagnify as LZShiftAndMagnify
from ._le_interpolation_lanczos import ShiftScaleRotate as LZShiftScaleRotate
from ._le_interpolation_nearest_neighbor import ShiftAndMagnify as NNShiftAndMagnify
from ._le_interpolation_nearest_neighbor import ShiftScaleRotate as NNShiftScaleRotate
from ._le_mandelbrot_benchmark import MandelbrotBenchmark
