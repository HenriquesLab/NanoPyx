def rebin_2d(arr, bin_factor, mode="sum"):
    """
    Bins a 2D array by a given factor. The last two dimensions of the array are binned.
    :param arr: numpy array with any shape as long as last two dimensions are y, x (example: time, channel, z, y, x)
    :param bin_factor: factor used to bin dimensions
    :param mode: can be either sum, mean or max, defaults to sum if not specified or not valid mode
    :return: binned array
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

    if mode == "sum":
        return arr.reshape(
            arr.shape[:-2]
            + (
                arr.shape[-2] // bin_factor,
                bin_factor,
                arr.shape[-1] // bin_factor,
                bin_factor,
            )
        ).sum(axis=(-1, -3))
    elif mode == "mean":
        return arr.reshape(
            arr.shape[:-2]
            + (
                arr.shape[-2] // bin_factor,
                bin_factor,
                arr.shape[-1] // bin_factor,
                bin_factor,
            )
        ).mean(axis=(-1, -3))
    elif mode == "max":
        return arr.reshape(
            arr.shape[:-2]
            + (
                arr.shape[-2] // bin_factor,
                bin_factor,
                arr.shape[-1] // bin_factor,
                bin_factor,
            )
        ).max(axis=(-1, -3))
