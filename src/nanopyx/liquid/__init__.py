import os
import warnings

from .__le_interpolation_catmull_rom import \
    ShiftAndMagnify as CRShiftAndMagnify
from .__le_interpolation_catmull_rom import \
    ShiftScaleRotate as CRShiftScaleRotate
from .__le_interpolation_nearest_neighbor import \
    ShiftAndMagnify as NNShiftAndMagnify
from .__le_interpolation_nearest_neighbor import \
    ShiftScaleRotate as NNShiftScaleRotate
from .__le_mandelbrot_benchmark import MandelbrotBenchmark
from .__njit__ import njit_works
from .__opencl__ import cl, cl_array, cl_ctx, cl_queue, opencl_works
