import numpy as np


def magnify(
    image: np.ndarray,
    magnification: float = 2,
    enforce_same_value: bool = True,
) -> np.ndarray:
    """
    Zoom an image by zero-padding its Discrete Fourier transform
    :param image: 2D grid of pixel values
    :param magnification: factor by which to multiply the dimensions of the image
    :param enforce_same_value: if True, the value of the original samples will be preserved
    :return: zoomed image

    REF: based on https://github.com/centreborelli/fourier
    """
    rows, cols = image.shape

    # Fourier transform with the zero-frequency component at the center
    imageFt = np.fft.fftshift(np.fft.fft2(image))

    # the zoom-in is performed by zero padding the Fourier transform
    rowsM = rows * magnification
    colsM = cols * magnification
    r0 = rowsM // 2 - rows // 2
    c0 = colsM // 2 - cols // 2
    imageFtPadded = np.zeros((rowsM, colsM), dtype=np.complex64)
    imageFtPadded[r0 : r0 + rows, c0 : c0 + cols] = imageFt

    # apply ifftshift before taking the inverse Fourier transform
    imageM = np.fft.ifft2(np.fft.ifftshift(imageFtPadded))

    # if the input is a real-valued image, then keep only the real part
    if np.isrealobj(image):
        imageM = np.real(imageM)

    # to preserve the values of the original samples, the L2 norm has to by multiplied by magnification*magnification
    imageM *= magnification * magnification

    if enforce_same_value:
        imageM[::magnification, ::magnification] = image

    # return the image casted to the input data type
    return imageM.astype(image.dtype, copy=False)
