"""
.. include:: ../../README.md
"""

import os

from . import _version, core, data, methods, liquid  # noqa: F401

__version__ = _version.get_versions()["version"]

# Get the user's home folder
__home_folder__ = os.path.expanduser("~")
__config_folder__ = os.path.join(__home_folder__, ".nanopyx")
if not os.path.exists(__config_folder__):
    os.makedirs(__config_folder__)
