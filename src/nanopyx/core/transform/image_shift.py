"""
Combination of functions for shifting an image, using several interpolation methods.
"""

from scipy.ndimage import shift
import numpy as np

from ..utils.timeit import timeit2
from . import interpolation_bicubic
from . import interpolation_bilinear
from . import interpolation_nearest_neighbor
from . import interpolation_catmull_rom
from . import interpolation_lanczos

from skimage.transform import AffineTransform, warp
import cv2


@timeit2
def catmull_rom_shift(image: np.ndarray, dx: float, dy: float):
    interpolator = interpolation_catmull_rom.Interpolator(image)
    return interpolator.shift(dx, dy)


@timeit2
def lanczos_shift(image: np.ndarray, dx: float, dy: float):
    interpolator = interpolation_lanczos.Interpolator(image)
    return interpolator.shift(dx, dy)


@timeit2
def bicubic_shift(image: np.ndarray, dx: float, dy: float):
    interpolator = interpolation_bicubic.Interpolator(image)
    return interpolator.shift(dx, dy)


@timeit2
def bilinear_shift(image: np.ndarray, dx: float, dy: float):
    interpolator = interpolation_bilinear.Interpolator(image)
    return interpolator.shift(dx, dy)


@timeit2
def nearest_neighbor_shift(image: np.ndarray, dx: float, dy: float):
    interpolator = interpolation_nearest_neighbor.Interpolator(image)
    return interpolator.shift(dx, dy)


@timeit2
def scipy_shift(image: np.ndarray, dx: float, dy: float):
    return shift(image, (dy, dx))


@timeit2
def skimage_shift(image: np.ndarray, dx: float, dy: float):
    tform = AffineTransform(translation=(-dx, -dy))
    return warp(image, tform)


@timeit2
def cv2_shift(image: np.ndarray, dx: float, dy: float):
    return cv2.warpAffine(
        image, np.float32([[1, 0, dx], [0, 1, dy]]), image.shape[:2][::-1]
    )
