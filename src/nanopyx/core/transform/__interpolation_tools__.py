import numpy as np


def check_image(image: np.ndarray) -> np.ndarray:
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
    if image.ndim != 2 and image.ndim != 3:
        raise ValueError("Image must be 2D and 3D (sequence of 2D images)")
    if image.dtype != np.float32:
        image = image.astype(np.float32, copy=False)
    if image.ndim == 2:
        image = image.reshape((1, image.shape[0], image.shape[1]))
    return image


def value2array(v, n_frames: int) -> np.ndarray:
    """
    Convert a value to an array of the same length as the number of frames
    :param v: value to convert
    :type v: int, float, np.ndarray
    :param n_frames: number of frames
    :return: array of the same length as the number of frames
    """
    if type(v) in (int, float):
        v = np.ones(n_frames, dtype=np.float32) * v
    return v
