"""
Combination of functions for zooming an image, using several interpolation methods.
"""

import numpy as np
from cv2 import INTER_LANCZOS4
from cv2 import resize as cv2_resize
from scipy.ndimage import zoom
from skimage.transform import rescale

from . import interpolation_fft_zoom

from ..utils.timeit import timeit2
from . import interpolation_bicubic
from . import interpolation_bilinear
from . import interpolation_nearest_neighbor
from . import interpolation_catmull_rom
from . import interpolation_lanczos


@timeit2
def fourier_zoom(image: np.ndarray, magnification: float = 2) -> np.ndarray:
    """
    Zoom an image by zero-padding its Discrete Fourier transform.
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :return: zoomed image.
    """

    return interpolation_fft_zoom.magnify(image, magnification)


@timeit2
def catmull_rom_zoom(image: np.ndarray, magnification: int = 2) -> np.ndarray:
    """
    Zoom an image by Catmull-Rom interpolation
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :return: zoomed image.
    REF: based on https://github.com/HenriquesLab/NanoJ-SRRF/blob/master/SRRF/src/nanoj/srrf/java/SRRF.java
    """

    interpolator = interpolation_catmull_rom.Interpolator(image)
    return interpolator.magnify(magnification)

def catmull_rom_zoom_xy(image: np.ndarray, magnification_y: int = 2, magnification_x: int = 2) -> np.ndarray:
    interpolator = interpolation_catmull_rom.Interpolator(image)
    return interpolator.magnify_xy(magnification_y, magnification_x)

@timeit2
def lanczos_zoom(
    image: np.ndarray, magnification: int = 2) -> np.ndarray:
    """
    Zoom an image by Lanczos interpolation
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :param taps: The number of taps (interpolation points) to use in the Lanczos kernel.
    :return: zoomed image.
    """

    interpolator = interpolation_lanczos.Interpolator(image)
    return interpolator.magnify(magnification)


@timeit2
def bicubic_zoom(image: np.ndarray, magnification: int = 2) -> np.ndarray:
    """
    Zoom an image by bicubic interpolation
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :return: zoomed image.
    """

    interpolator = interpolation_bicubic.Interpolator(image)
    return interpolator.magnify(magnification)


@timeit2
def bilinear_zoom(image: np.ndarray, magnification: int = 2) -> np.ndarray:
    """
    Zoom an image by bilinear interpolation
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :return: zoomed image.
    """

    interpolator = interpolation_bilinear.Interpolator(image)
    return interpolator.magnify(magnification)


@timeit2
def nearest_neighbor_zoom(
    image: np.ndarray, magnification: int = 2
) -> np.ndarray:
    """
    Zoom an image by nearest neighbor interpolation
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :return: zoomed image.
    """

    interpolator = interpolation_nearest_neighbor.Interpolator(image)
    return interpolator.magnify(magnification)


@timeit2
def scipy_zoom(image: np.ndarray, magnification: int = 2) -> np.ndarray:
    """
    Zoom an image by SciPy interpolation
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :return: zoomed image.
    """

    return zoom(image, magnification)


@timeit2
def skimage_zoom(image: np.ndarray, magnification: int = 2) -> np.ndarray:
    """
    Zoom an image by scikit-image interpolation
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :return: zoomed image.
    """

    return rescale(image, magnification, anti_aliasing=False)


@timeit2
def cv2_zoom(image: np.ndarray, magnification: int = 2) -> np.ndarray:
    """
    Zoom an image by OpenCV interpolation
    :param image: 2D grid of pixel values.
    :param magnification: Factor by which to multiply the dimensions of the image.
    :return: zoomed image.
    """

    return cv2_resize(
        image,
        None,
        fx=magnification,
        fy=magnification,
        interpolation=INTER_LANCZOS4,
    )
