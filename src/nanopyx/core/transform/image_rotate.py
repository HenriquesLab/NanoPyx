"""
Combination of functions for rotating an image, using several interpolation methods.
"""

from scipy.ndimage import rotate
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
def catmull_rom_rotate(image: np.ndarray, angle:float):
    interpolator = interpolation_catmull_rom.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def lanczos_rotate(image: np.ndarray, angle:float):
    interpolator = interpolation_lanczos.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def bicubic_rotate(image: np.ndarray, angle:float):
    interpolator = interpolation_bicubic.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def bilinear_rotate(image: np.ndarray, angle:float):
    interpolator = interpolation_bilinear.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def nearest_neighbor_rotate(image: np.ndarray, angle:float):
    interpolator = interpolation_nearest_neighbor.Interpolator(image)
    return interpolator.rotate(angle)


@timeit2
def scipy_rotate(image: np.ndarray, angle:float):
    return rotate(image, np.rad2deg(angle), reshape=False)


@timeit2
def skimage_rotate(image: np.ndarray, angle:float):
    rotation = AffineTransform(rotation=angle)
    shift2center = AffineTransform(translation=-np.array(image.shape[:2][::-1]) / 2)
    shift2og = AffineTransform(translation=np.array(image.shape[:2][::-1]) / 2)
    return warp(image, shift2center + rotation + shift2og)


@timeit2
def cv2_rotate(image: np.ndarray, angle:float):
    rotmat = cv2.getRotationMatrix2D(np.array(image.shape[:2][::-1]) / 2, np.rad2deg(angle), 1)
    return cv2.warpAffine(image, rotmat, image.shape[:2][::-1])
