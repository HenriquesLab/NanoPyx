"""
.. include:: ../../README.md
"""

import os
import pkg_resources  # part of setuptools

__version__ = pkg_resources.require("NanoPyx")[0].version

from . import core, data, methods  # noqa: F401

# Get the user's home folder
__home_folder__ = os.path.expanduser("~")
__config_folder__ = os.path.join(__home_folder__, ".nanopyx")
if not os.path.exists(__config_folder__):
    os.makedirs(__config_folder__)

from .__agent__ import Agent  # noqa: E402

# TODO: allow benchmarking of only specific implementations
# TODO: provide parallelized batch processing

from .__njit__ import njit_works
from .__opencl__ import cl, cl_array, opencl_works, print_opencl_info, devices

from .core.utils.benchmark import benchmark_all_le_methods as benchmark

__all__ = [
    "core",
    "data",
    "methods",
    "__liquid_engine__",
    "__agent__",
    "__cuda__",
    "__dask__",
    "__njit__",
    "__transonic__",
    "__opencl__",
]

# Section for imports of high-level functions
from .core.utils.benchmark import benchmark_all_le_methods as benchmark
from .methods import non_local_means_denoising
from .methods import eSRRF, run_esrrf_parameter_sweep, eSRRF3D
from .methods import SRRF
from .methods import calculate_frc, calculate_decorr_analysis
from .methods import calculate_error_map
from .methods import estimate_drift_alignment, apply_drift_alignment
from .methods import estimate_channel_registration, apply_channel_registration
