import numpy as np

from nanopyx.core.transform.interpolation.catmull_rom import (
    magnify as _catmull_rom_magnify,
)

# from scipy.stats import \
#    linregress  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html


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

    # Fourier transform with the zero-frequency component at the center
    imageFt = np.fft.fftshift(np.fft.fft2(image))

    # the zoom-in is performed by zero padding the Fourier transform
    wM = w * magnification
    hM = h * magnification
    x0 = wM // 2 - w // 2
    y0 = hM // 2 - h // 2
    imageFtPadded = np.zeros((wM, hM), dtype=np.complex64)
    imageFtPadded[x0 : x0 + w, y0 : y0 + h] = imageFt

    # apply ifftshift before taking the inverse Fourier transform
    imageM = np.fft.ifft2(np.fft.ifftshift(imageFtPadded))

    # if the input is a real-valued image, then keep only the real part
    if np.isrealobj(image):
        imageM = np.real(imageM)

    # to preserve the values of the original samples, the L2 norm has to by multiplied by magnification*magnification
    imageM *= magnification * magnification

    # slope, intercept, r, p, se = linregress(
    #     image.ravel(), imageM[::magnification, ::magnification].ravel()
    # )
    # print(slope, intercept, r, p, se)

    # return the image casted to the input data type
    return imageM.astype(image.dtype)


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
