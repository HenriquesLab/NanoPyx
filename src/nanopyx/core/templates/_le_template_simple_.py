import numpy as np

from ...__njit__ import njit, prange


def py_template(image: np.ndarray) -> np.ndarray:
    """
    Template python function
    :param image:
    """
    for p in range(image.shape[0]):
        pass

    return np.asarray(image)


@njit(cache=True, parallel=True)
def njit_template(image: np.ndarray) -> np.ndarray:
    """
    Template numba accelerated python function
    :param image:
    """
    for f in prange(image.shape[0]):
        pass

    return np.asarray(image)
