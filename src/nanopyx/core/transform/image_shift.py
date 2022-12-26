from scipy.ndimage.interpolation import shift
import numpy as np

def shift_t2d(im: np.ndarray, dx: np.ndarray, dy: np.ndarray):
    assert im.ndim == 3
    assert dx.ndim == 1
    assert dy.ndim == 1
    assert dx.shape[0] == dy.shape[0]

    deltas = np.vstack((np.zeros_like(dx), dx, dy)) # mode='nearest'

    return shift(im, deltas)

