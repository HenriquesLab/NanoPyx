import numpy as np
from cv2 import INTER_LANCZOS4
from cv2 import resize as cv2_resize
from scipy.ndimage import zoom
from skimage.transform import rescale

from ..utils.time.timeit import timeit2
from .interpolation import bicubic, catmull_rom, fft_zoom, lanczos

# from scipy.stats import \
#    linregress  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html


@timeit2
def fourier_zoom(image: np.ndarray, magnification: float = 2):
    """
    Zoom an image by zero-padding its Discrete Fourier transform.

    Args:
        image (np.ndarray): 2D grid of pixel values.
        magnification (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.

    Returns:
        np.ndarray: zoomed image.

    REF: based on https://github.com/centreborelli/fourier
    Credit goes to Carlo de Franchis <carlo.de-franchis@ens-paris-saclay.fr>
    """
    return fft_zoom.magnify(image, magnification)


@timeit2
def catmull_rom_zoom(image: np.ndarray, magnification: int = 2):
    """
    Zoom an image by Catmull-Rom interpolation

    Args:
        image (np.ndarray): 2D grid of pixel values.
        magnification (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.

    Returns:
        np.ndarray: zoomed image.

    REF: based on https://github.com/HenriquesLab/NanoJ-SRRF/blob/master/SRRF/src/nanoj/srrf/java/SRRF.java
    """
    return catmull_rom.magnify(image, magnification)


@timeit2
def lanczos_zoom(image: np.ndarray, magnification: int = 2, taps: int = 3):
    """
    Zoom an image by Lanczos interpolation

    Args:
        image (np.ndarray): 2D grid of pixel values.
        magnification (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.
        taps (int): The number of taps (interpolation points) to use in the Lanczos kernel.

    Returns:
        np.ndarray: zoomed image.
    """
    return lanczos.magnify(image, magnification, taps)


@timeit2
def bicubic_zoom(image: np.ndarray, magnification: int = 2):
    """
    Zoom an image by bicubic interpolation

    Args:
        image (np.ndarray): 2D grid of pixel values.
        magnification (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.

    Returns:
        np.ndarray: zoomed image.

    """
    return bicubic.magnify(image, magnification)


@timeit2
def scipy_zoom(image: np.ndarray, magnification: int = 2):
    """
    Zoom an image by SciPy interpolation

    Args:
        image (np.ndarray): 2D grid of pixel values.
        magnification (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.

    Returns:
        np.ndarray: zoomed image.

    """

    return zoom(image, magnification)


@timeit2
def skimage_zoom(image: np.ndarray, magnification: int = 2):
    """
    Zoom an image by scikit-image interpolation

    Args:
        image (np.ndarray): 2D grid of pixel values.
        magnification (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.

    Returns:
        np.ndarray: zoomed image.

    """

    return rescale(image, magnification, anti_aliasing=False)


@timeit2
def cv2_zoom(image: np.ndarray, magnification: int = 2):
    """
    Zoom an image by OpenCV interpolation

    Args:
        image (np.ndarray): 2D grid of pixel values.
        magnification (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.

    Returns:
        np.ndarray: zoomed image.

    """

    return cv2_resize(
        image, None, fx=magnification, fy=magnification, interpolation=INTER_LANCZOS4
    )
