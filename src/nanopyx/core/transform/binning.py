import numpy as np

def rebin_2d(arr: np.ndarray, bin_factor: int, mode: str = "sum") -> np.ndarray:
    """
    Bins a 2D array by a given factor. The last two dimensions of the array are binned.

    :param arr: numpy array with any shape as long as last two dimensions are y, x (example: time, channel, z, y, x)
    :type arr: numpy.ndarray
    :param bin_factor: factor used to bin dimensions
    :type bin_factor: int
    :param mode: can be either sum, mean or max, defaults to sum if not specified or not valid mode
    :type mode: str
    :return: binned array
    :rtype: numpy.ndarray
    """
    if mode not in ["sum", "mean", "max"]:
        mode = "sum"

    if arr.ndim < 2:
        raise ValueError("Array must have at least 2 dimensions")

    if bin_factor == 1:
        return arr

    if bin_factor < 1:
        raise ValueError("Binning factor must be greater than 1")

    if arr.shape[-1] % bin_factor != 0 or arr.shape[-2] % bin_factor != 0:
        raise ValueError("Binning factor must be a divisor of the last two dimensions")

    reshaped_arr = arr.reshape(
        arr.shape[:-2]
        + (
            arr.shape[-2] // bin_factor,
            bin_factor,
            arr.shape[-1] // bin_factor,
            bin_factor,
        )
    )

    if mode == "sum":
        return reshaped_arr.sum(axis=(-1, -3))
    elif mode == "mean":
        return reshaped_arr.mean(axis=(-1, -3))
    else:
        return reshaped_arr.max(axis=(-1, -3))
