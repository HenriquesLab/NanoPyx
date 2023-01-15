from scipy.ndimage import shift
import numpy as np

from ..utils.time.timeit import timeit2
from .interpolation import bicubic, bilinear, catmull_rom, lanczos

from skimage.transform import AffineTransform, warp
import cv2


@timeit2
def catmull_rom_shift(image: np.ndarray, dx: float, dy: float):
    interpolator = catmull_rom.Interpolator(image)
    return interpolator.shift(dx, dy)


@timeit2
def lanczos_shift(image: np.ndarray, dx: float, dy: float, taps: int = 3):
    interpolator = lanczos.Interpolator(image, taps)
    return interpolator.shift(dx, dy)


@timeit2
def bicubic_shift(image: np.ndarray, dx: float, dy: float):
    interpolator = bicubic.Interpolator(image)
    return interpolator.shift(dx, dy)


@timeit2
def bilinear_shift(image: np.ndarray, dx: float, dy: float):
    interpolator = bilinear.Interpolator(image)
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
