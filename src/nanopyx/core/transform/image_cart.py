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

@timeit2
def catmull_rom_cart(image: np.ndarray, x_shape: int, y_shape: int, scale:str='linear'):
    interpolator = interpolation_catmull_rom.Interpolator(image)
    return interpolator.cartesian(x_shape,y_shape,scale)


@timeit2
def lanczos_cart(image: np.ndarray, x_shape: int, y_shape: int, scale:str='linear'):
    interpolator = interpolation_lanczos.Interpolator(image)
    return interpolator.cartesian(x_shape,y_shape,scale)


@timeit2
def bicubic_cart(image: np.ndarray, x_shape: int, y_shape: int, scale:str='linear'):
    interpolator = interpolation_bicubic.Interpolator(image)
    return interpolator.cartesian(x_shape,y_shape,scale)


@timeit2
def bilinear_cart(image: np.ndarray, x_shape: int, y_shape: int, scale:str='linear'):
    interpolator = interpolation_bilinear.Interpolator(image)
    return interpolator.cartesian(x_shape,y_shape,scale)


@timeit2
def nearest_neighbor_cart(image: np.ndarray, x_shape: int, y_shape: int, scale:str='linear'):
    interpolator = interpolation_nearest_neighbor.Interpolator(image)
    return interpolator.cartesian(x_shape,y_shape,scale)



