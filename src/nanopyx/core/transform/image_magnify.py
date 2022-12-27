import numpy as np
from nanopyx.core.transform.interpolation.catmull_rom import (
    magnify as _catmull_rom_magnify,
)


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
    w, h = image.shape

    # zero padding sizes
    left = np.ceil((magnification - 1) * h / 2).astype(int)
    right = np.floor((magnification - 1) * h / 2).astype(int)
    top = np.ceil((magnification - 1) * w / 2).astype(int)
    bottom = np.floor((magnification - 1) * w / 2).astype(int)

    # Fourier transform with the zero-frequency component at the center
    ft = np.fft.fftshift(np.fft.fft2(image))

    # the zoom-in is performed by zero padding the Fourier transform
    ft = np.pad(ft, [(top, bottom), (left, right)], "constant", constant_values=0 + 0j)

    # apply ifftshift before taking the inverse Fourier transform
    out = np.fft.ifft2(np.fft.ifftshift(ft))

    # if the input is a real-valued image, then keep only the real part
    if np.isrealobj(image):
        out = np.real(out)

    # to preserve the values of the original samples, the L2 norm has to be
    # multiplied by z*z.
    out *= magnification * magnification

    # clip values to avoid overflow when casting to input dtype
    if np.issubdtype(image.dtype, np.integer):
        i = np.iinfo(image.dtype)
        out = np.round(out)
    else:
        i = np.finfo(image.dtype)
    out = np.clip(out, i.min, i.max)

    # return the image casted to the input data type
    return out.astype(image.dtype)


def catmull_rom_zoom(image: np.ndarray, magnification: int = 2):
    """
    Zoom an image by Catmull-Rom interpolation

    Args:
        image (np.ndarray): 2D grid of pixel values.
        z (float): Factor by which to multiply the dimensions of the image.
            Must be >= 1.

    Returns:
        np.ndarray: zoomed image.

    REF: based on https://github.com/HenriquesLab/NanoJ-SRRF/blob/master/SRRF/src/nanoj/srrf/java/SRRF.java
    """
    return _catmull_rom_magnify(image, magnification)
