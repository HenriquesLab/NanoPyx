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
def catmull_rom_rotate(image: np.ndarray, angle:float):
    interpolator = catmull_rom.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def lanczos_rotate(image: np.ndarray, angle:float, taps: int = 3):
    interpolator = lanczos.Interpolator(image, taps)
    return interpolator.rotate(angle)


@timeit2
def bicubic_rotate(image: np.ndarray, angle:float):
    interpolator = bicubic.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def bilinear_rotate(image: np.ndarray, angle:float):
    interpolator = bilinear.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def nearest_neighbor_rotate(image: np.ndarray, angle:float):
    interpolator = nearest_neighbor.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def scipy_rotate(image: np.ndarray, angle:float):
    return rotate(image, angle)


@timeit2
def skimage_rotate(image: np.ndarray, angle:float):
    tform = AffineTransform(rotation=angle)
    return warp(image, tform)


@timeit2
def cv2_rotate(image: np.ndarray, angle:float, cx: float, cy: float):
    return cv2.warpAffine(
        image, np.float32([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0]]), image.shape[:2][::-1]
    )
