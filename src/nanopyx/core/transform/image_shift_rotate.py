"""
Combination of functions for rotating an image over an arbitrary center, using several interpolation methods.
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
def catmull_rom_rotate_shift(image: np.ndarray, angle:float, cx:float, cy:float):
    interpolator = catmull_rom.Interpolator(image)
    return interpolator.rotate(angle, cx, cy)


@timeit2
def lanczos_rotate_shift(image: np.ndarray, angle:float, cx:float, cy:float, taps: int = 3):
    interpolator = lanczos.Interpolator(image, taps)
    return interpolator.rotate(angle, cx, cy)


@timeit2
def bicubic_rotate_shift(image: np.ndarray, angle:float, cx:float, cy:float):
    interpolator = bicubic.Interpolator(image)
    return interpolator.rotate(angle, cx, cy)


@timeit2
def bilinear_rotate_shift(image: np.ndarray, angle:float, cx:float, cy:float):
    interpolator = bilinear.Interpolator(image)
    return interpolator.rotate(angle, cx, cy)


@timeit2
def nearest_neighbor_rotate_shift(image: np.ndarray, angle:float, cx:float, cy:float):
    interpolator = nearest_neighbor.Interpolator(image)
    return interpolator.rotate(angle, cx, cy)


#def scipy_rotate_shift(image: np.ndarray, angle:float, cx:float, cy:float):
#scipy does not have a simple rotation around an arbitrary point which preserves image size
#shift + rotation + shift would work but we lose information

@timeit2
def skimage_rotate_shift(image: np.ndarray, angle:float, cx:float, cy:float):
    rotation = AffineTransform(rotation=angle)
    shift2center = AffineTransform(translation=-np.array([cx, cy]))
    shift2og = AffineTransform(translation=np.array([cx, cy]))
    return warp(image, shift2center + rotation + shift2og)


@timeit2
def cv2_rotate_shift(image: np.ndarray, angle:float, cx:float, cy:float):
    rotmat = cv2.getRotationMatrix2D(np.array([cx,cy]), np.rad2deg(angle), 1)
    return cv2.warpAffine(image, rotmat, image.shape[:2][::-1])
