"""
.. include:: ../../README.md
"""

import os

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
