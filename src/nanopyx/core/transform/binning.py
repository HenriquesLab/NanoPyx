

def rebin2d(arr, bin_factor, mode="sum"):
    """
    :param arr: numpy array with any shape as long as last two dimensions are y, x (example: time, channel, z, y, x)
    :param bin_factor: factor used to bin dimensions
    :param mode: can be either sum, mean or max, defaults to sum if not specified or not valid mode
    :return: binned array
    """
    new_shape = [int(arr.shape[-2] / bin_factor), int(arr.shape[-1] / bin_factor)]
    shape = (new_shape[-2], arr.shape[-2] // new_shape[-2],
             new_shape[-1], arr.shape[-1] // new_shape[-1])
    if mode == "mean":
        return arr.reshape(shape).mean(-1).mean(1)
    elif mode == "max":
        return arr.reshape(shape).max(-1).max(1)
    else:
        return arr.reshape(shape).sum(-1).sum(1)
