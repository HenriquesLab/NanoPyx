"""
The liquid module contains the core functionality of the liquid engine.

The liquid engine is a NanoPyx library specialized in adaptive processing, where the member classes are able to
benchmark themselves and compare their performance for several implementations of the same functionality (OpenCL,
Cython, Numba, etc.).

>>> from nanopyx.liquid import MandelbrotBenchmark
>>> mb = MandelbrotBenchmark()
>>> values = mb.benchmark(64) # doctest: +SKIP
"""

import os  # noqa: F401
import warnings  # noqa: F401

from ._le_interpolation_catmull_rom import (  # noqa: F401
    ShiftAndMagnify as CRShiftAndMagnify,
)
from ._le_interpolation_catmull_rom import (  # noqa: F401
    ShiftScaleRotate as CRShiftScaleRotate,
)
from ._le_interpolation_nearest_neighbor import (  # noqa: F401
    ShiftAndMagnify as NNShiftAndMagnify,
)
from ._le_interpolation_nearest_neighbor import (  # noqa: F401
    ShiftScaleRotate as NNShiftScaleRotate,
)
from ._le_mandelbrot_benchmark import MandelbrotBenchmark  # noqa: F401
from .__njit__ import njit_works  # noqa: F401
from .__opencl__ import (  # noqa: F401
    cl,
    cl_array,
    cl_ctx,
    cl_queue,
    opencl_works,
)
