"""
Combination of functions for rotating an image, using several interpolation methods.
"""

from scipy.ndimage import rotate
import numpy as np

from ..utils.time.timeit import timeit2
from .interpolation import (
    bicubic,
    bilinear,
    nearest_neighbor,
    catmull_rom,
    lanczos,
)

from skimage.transform import AffineTransform, warp
import cv2

@timeit2
def catmull_rom_shift(image: np.ndarray, angle:float, cx: float, cy: float):
    interpolator = catmull_rom.Interpolator(image)
    return interpolator.rotate(angle, cx, cy)


@timeit2
def lanczos_shift(image: np.ndarray, angle:float, cx: float, cy: float, taps: int = 3):
    interpolator = lanczos.Interpolator(image, taps)
    return interpolator.shift(angle, cx, cy)


@timeit2
def bicubic_shift(image: np.ndarray, angle:float, cx: float, cy: float):
    interpolator = bicubic.Interpolator(image)
    return interpolator.shift(angle, cx, cy)


@timeit2
def bilinear_shift(image: np.ndarray, angle:float, cx: float, cy: float):
    interpolator = bilinear.Interpolator(image)
    return interpolator.shift(angle, cx, cy)


@timeit2
def nearest_neighbor_shift(image: np.ndarray, angle:float, cx: float, cy: float):
    interpolator = nearest_neighbor.Interpolator(image)
    return interpolator.shift(angle, cx, cy)


@timeit2
def scipy_shift(image: np.ndarray, angle:float, cx: float, cy: float):
    return shift(image, angle)


@timeit2
def skimage_shift(image: np.ndarray, angle:float, cx: float, cy: float):
    tform = AffineTransform(translation=(-dx, -dy))
    return warp(image, tform)


@timeit2
def cv2_shift(image: np.ndarray, angle:float, cx: float, cy: float):
    return cv2.warpAffine(
        image, np.float32([[1, 0, dx], [0, 1, dy]]), image.shape[:2][::-1]
    )
