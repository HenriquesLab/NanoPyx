import numpy as np


def check_image(image: np.ndarray) -> np.ndarray:
    if type(image) is not np.ndarray:
        raise TypeError("Image must be of type np.ndarray")
    if image.ndim != 2 and image.ndim != 3:
        raise ValueError("Image must be 2D and 3D (sequence of 2D images)")
    if image.dtype != np.float32:
        raise TypeError("Image must be of type np.float32")
    if image.ndim == 2:
        image = image.reshape((1, image.shape[0], image.shape[1])).astype(
            np.float32, copy=False
        )
    return image


def value2array(shift: np.ndarray | int | float, n_frames: int) -> np.ndarray:
    if type(shift) in (int, float):
        shift = np.ones(n_frames, dtype=np.float32) * shift
    return shift
