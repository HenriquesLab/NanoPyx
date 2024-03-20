import numpy as np
from ...core.transform.error_map import ErrorMap


def calculate_error_map(img_ref: np.ndarray, img_sr: np.ndarray):
    """
    Calculate the error map between a reference image and a super-resolved image.

    This function utilizes the ErrorMap class to compute the error map between
    the provided reference image (`img_ref`) and the super-resolved image (`img_sr`).
    It optimizes the parameters to minimize the difference between the scaled,
    blurred version of the super-resolved image and the reference image, and
    returns the resulting error map, the root square error (RSE), and the
    root square Pearson correlation (RSP).

    Parameters
    ----------
    img_ref : np.ndarray
        The reference image against which the super-resolved image is compared.
        Expected to be a 2D numpy array.
    img_sr : np.ndarray
        The super-resolved image that is being evaluated.
        Expected to be a 2D numpy array.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - np.ndarray: The error map as a 2D numpy array of type np.float32.
        - float: The root square error (RSE) value.
        - float: The root square Pearson correlation (RSP) value.
    """

    emc = ErrorMap()
    emc.optimise(img_ref, img_sr)
    return np.asarray(emc.imRSE, dtype=np.float32), emc.getRSE(), emc.getRSP()