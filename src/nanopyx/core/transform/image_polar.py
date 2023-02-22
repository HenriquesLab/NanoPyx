"""
Combination of functions for shifting an image, using several interpolation methods.
"""

import numpy as np

from ..utils.time.timeit import timeit2
from .interpolation import (
    bicubic,
    bilinear,
    nearest_neighbor,
    catmull_rom,
    lanczos,
)

from skimage.transform import warp_polar

@timeit2
def catmull_rom_polar(image: np.ndarray, scale:str='linear'):
    interpolator = catmull_rom.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def lanczos_polar(image: np.ndarray, scale:str='linear', taps: int = 3):
    interpolator = lanczos.Interpolator(image, taps)
    return interpolator.polar(scale)


@timeit2
def bicubic_polar(image: np.ndarray, scale:str='linear'):
    interpolator = bicubic.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def bilinear_polar(image: np.ndarray, scale:str='linear'):
    interpolator = bilinear.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def nearest_neighbor_polar(image: np.ndarray, scale:str='linear'):
    interpolator = nearest_neighbor.Interpolator(image)
    return interpolator.polar(scale)


@timeit2
def skimage_polar(image: np.ndarray, scale:str='linear'):
    return warp_polar(image, scaling=scale)


