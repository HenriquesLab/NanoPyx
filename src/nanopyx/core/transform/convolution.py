import warnings
import numpy as np


try:
    from numba import njit
except ImportError:
    print("Optional dependency Numba is not installed. Numba implementations will be ignored.")

    def njit(*args, **kwargs):
        def wrapper(func):
            warnings.warn(f"Numba is not installed. Using pure python for {func.__name__}")
            return func

        return wrapper

try:
    import cupy as cp
    from cupyx.scipy.signal import convolve2d as cupyx_convolve
except ImportError:
    print("Cupy implementation is not available. Make sure you have the right version of Cupy and CUDA installed.")

try:
    import dask.array as da

except ImportError:
    print("Optional dependency Dask is not installed. Dask implementations will be ignored.")

try:
    from dask_image.ndfilters import convolve as dask_convolve
except ImportError:
    print("Optional dependecy Dask_image is not installed. Implementations using it will be ignored.")


# def check_array(image: np.ndarray):
#     """
#     Check the given image and ensure it meets the required conditions.

#     Parameters:
#         image (numpy.ndarray): The image to be checked.

#     Returns:
#         numpy.ndarray: The checked and potentially modified image.

#     Raises:
#         TypeError: If the image is not of type numpy.ndarray.
#         ValueError: If the image is not 2D or 3D.
#     """
#     image = np.asarray(image)
#     if type(image) is not np.ndarray:
#         raise TypeError("Image must be of type np.ndarray")
#     if image.ndim != 3:
#         raise ValueError("Image must be 2D")
#     if image.dtype != np.float32:
#         image = image.astype(np.float32, copy=False)
#     return image


def convolution2D_python(image: np.ndarray, kernel: np.ndarray):
    nFrames = image.shape[0]
    nRows = image.shape[1]
    nCols = image.shape[2]

    nRows_kernel = kernel.shape[0]
    nCols_kernel = kernel.shape[1]

    center_r = (nRows_kernel-1) // 2
    center_c = (nCols_kernel-1) // 2

    acc = 0.0

    conv_out = np.zeros((nFrames, nRows, nCols), dtype=np.float32)
    
    for f in range(nFrames):
        for r in range(nRows):
            for c in range(nCols):
                acc = 0
                for kr in range(nRows_kernel):
                    for kc in range(nCols_kernel):
                        local_row = min(max(r+(kr-center_r), 0), nRows-1)
                        local_col = min(max(c+(kc-center_c), 0), nCols-1)
                        acc = acc + kernel[kr, kc] * image[f,local_row, local_col]
                conv_out[f,r, c] = acc

    return conv_out


def convolution2D_transonic(image, kernel):
    try:
        import transonic
        from ._transonic import convolution2D
        return convolution2D(image, kernel)
    except ModuleNotFoundError:
        print("Transonic is not installed, defaulting to Python")
        return convolution2D_python(image, kernel)
    except ImportError:
        print("Transonic is not installed, defaulting to Python")
        return convolution2D_python(image, kernel)


@njit(cache=True, parallel=True)
def convolution2D_numba(image, kernel):
    nFrames = image.shape[0]
    nRows = image.shape[1]
    nCols = image.shape[2]

    nRows_kernel = kernel.shape[0]
    nCols_kernel = kernel.shape[1]

    center_r = (nRows_kernel-1) // 2
    center_c = (nCols_kernel-1) // 2

    acc = 0.0

    conv_out = np.zeros((nFrames, nRows, nCols), dtype=np.float32)

    for f in range(nFrames):
        for r in range(nRows):
            for c in range(nCols):
                acc = 0
                for kr in range(nRows_kernel):
                    for kc in range(nCols_kernel):
                        local_row = min(max(r+(kr-center_r), 0), nRows-1)
                        local_col = min(max(c+(kc-center_c), 0), nCols-1)
                        acc = acc + kernel[kr, kc] * image[f,local_row, local_col]
                conv_out[f,r, c] = acc

    return conv_out


def convolution2D_dask(image, kernel):

    conv_out = np.zeros_like(image)
    for i in range(image.shape[0]):
        conv_out[i] = dask_convolve(da.from_array(image[i]), da.from_array(kernel))
    return conv_out


def convolution2D_cuda(image, kernel):
    with cp.cuda.Device(0):
        output = cp.asnumpy(cupyx_convolve(cp.asarray(image), cp.asarray(kernel), mode="same", boundary="symm"))
    return output
