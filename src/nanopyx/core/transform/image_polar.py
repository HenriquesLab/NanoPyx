"""
Combination of functions for shifting an image, using several interpolation methods.
"""

import numpy as np

from ..utils.timeit import timeit2
from . import interpolation_bicubic
from . import interpolation_bilinear
from . import interpolation_nearest_neighbor
from . import interpolation_catmull_rom
from . import interpolation_lanczos

from skimage.transform import warp_polar

@timeit2
def catmull_rom_polar(image: np.ndarray, scale:str='linear'):
    interpolator = interpolation_catmull_rom.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def lanczos_polar(image: np.ndarray, scale:str='linear'):
    interpolator = interpolation_lanczos.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def bicubic_polar(image: np.ndarray, scale:str='linear'):
    interpolator = interpolation_bicubic.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def bilinear_polar(image: np.ndarray, scale:str='linear'):
    interpolator = interpolation_bilinear.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def nearest_neighbor_polar(image: np.ndarray, scale:str='linear'):
    interpolator = interpolation_nearest_neighbor.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def skimage_polar(image: np.ndarray, scale:str='linear'):
    return warp_polar(image, scaling=scale)


