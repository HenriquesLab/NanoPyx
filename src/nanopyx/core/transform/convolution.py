import transonic
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.signal import convolve2d as cupyx_convolve
from dask_image.ndfilters import convolve as dask_convolve
from numba import njit
from transonic import jit, boost


def check_array(image: np.ndarray):
    """
    Check the given image and ensure it meets the required conditions.

    Parameters:
        image (numpy.ndarray): The image to be checked.

    Returns:
        numpy.ndarray: The checked and potentially modified image.

    Raises:
        TypeError: If the image is not of type numpy.ndarray.
        ValueError: If the image is not 2D or 3D.
    """
    image = np.asarray(image)
    if type(image) is not np.ndarray:
        raise TypeError("Image must be of type np.ndarray")
    if image.ndim != 2:
        raise ValueError("Image must be 2D")
    if image.dtype != np.float32:
        image = image.astype(np.float32, copy=False)
    return image


def convolution2D_python(image: np.ndarray, kernel: np.ndarray):
    nRows = image.shape[0]
    nCols = image.shape[1]

    nRows_kernel = kernel.shape[0]
    nCols_kernel = kernel.shape[1]

    center_r = (nRows_kernel-1) // 2
    center_c = (nCols_kernel-1) // 2

    acc = 0.0

    conv_out = np.zeros((nRows, nCols), dtype=np.float32)

    for r in range(nRows):
        for c in range(nCols):
            acc = 0
            for kr in range(nRows_kernel):
                for kc in range(nCols_kernel):
                    local_row = min(max(r+(kr-center_r), 0), nRows-1)
                    local_col = min(max(c+(kc-center_c), 0), nCols-1)
                    acc = acc + kernel[kr, kc] * image[local_row, local_col]
            conv_out[r, c] = acc

    return conv_out


@jit(backend="numba")
def convolution2D_transonic(image: "float[]", kernel: "float[]"):
    nRows = image.shape[0]
    nCols = image.shape[1]

    nRows_kernel = kernel.shape[0]
    nCols_kernel = kernel.shape[1]

    center_r = (nRows_kernel-1) // 2
    center_c = (nCols_kernel-1) // 2

    acc = 0.0

    conv_out = np.zeros((nRows, nCols), dtype=np.float32)

    for r in range(nRows):
        for c in range(nCols):
            acc = 0
            for kr in range(nRows_kernel):
                for kc in range(nCols_kernel):
                    local_row = min(max(r+(kr-center_r), 0), nRows-1)
                    local_col = min(max(c+(kc-center_c), 0), nCols-1)
                    acc = acc + kernel[kr, kc] * image[local_row, local_col]
            conv_out[r, c] = acc

    return conv_out


@njit(cache=True, parallel=True)
def convolution2D_numba(image, kernel):
    nRows = image.shape[0]
    nCols = image.shape[1]

    nRows_kernel = kernel.shape[0]
    nCols_kernel = kernel.shape[1]

    center_r = (nRows_kernel-1) // 2
    center_c = (nCols_kernel-1) // 2

    acc = 0.0

    conv_out = np.zeros((nRows, nCols), dtype=np.float32)

    for r in range(nRows):
        for c in range(nCols):
            acc = 0
            for kr in range(nRows_kernel):
                for kc in range(nCols_kernel):
                    local_row = min(max(r+(kr-center_r), 0), nRows-1)
                    local_col = min(max(c+(kc-center_c), 0), nCols-1)
                    acc = acc + kernel[kr, kc] * image[local_row, local_col]
            conv_out[r, c] = acc

    return conv_out


def convolution2D_dask(image, kernel):
    return np.asarray(dask_convolve(da.from_array(image), da.from_array(kernel)))


def convolution2D_cuda(image, kernel):
    with cp.cuda.Device(0):
        output = cp.asnumpy(cupyx_convolve(cp.asarray(image), cp.asarray(kernel), mode="same", boundary="symm"))
    return output
