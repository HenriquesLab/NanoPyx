import numpy as np
from math import floor
from ...__njit__ import njit, prange


def _cubic(v):
    a = 0.5
    z = 0
    if v < 0:
        v = -v

    if v < 1:
        z = v * v * (v * (-a + 2) + (a - 3)) + 1
    elif v < 2:
        z = -a * v * v * v + 5 * a * v * v - 8 * a * v + 4 * a

    return z


@njit(cache=True)
def _njit_cubic(v):
    a = 0.5
    z = 0
    if v < 0:
        v = -v

    if v < 1:
        z = v * v * (v * (-a + 2) + (a - 3)) + 1
    elif v < 2:
        z = -a * v * v * v + 5 * a * v * v - 8 * a * v + 4 * a

    return z


def _interpolate(image, r, c, rows, cols):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return 0
    r_int = int(floor(r - 0.5))
    c_int = int(floor(c - 0.5))
    q = 0
    p = 0

    for j in range(4):
        c_neighbor = c_int - 1 + j
        p = 0
        if c_neighbor < 0 or c_neighbor >= cols:
            continue

        for i in range(4):
            r_neighbor = r_int - 1 + i
            if r_neighbor < 0 or r_neighbor >= rows:
                continue

            p = p + image[r_neighbor, c_neighbor] * _cubic(r - (r_neighbor + 0.5))
        q = q + p * _cubic(c - (c_neighbor + 0.5))

    return q


@njit(cache=True)
def _njit_interpolate(image, r, c, rows, cols):
    """
    A function that performs interpolation on an image using the catmull-rom method.

    Parameters:
    - image: numpy array representing the image
    - r: float, the row position of the pixel to interpolate
    - c: float, the column position of the pixel to interpolate
    - rows: int, the total number of rows in the image
    - cols: int, the total number of columns in the image

    Returns:
    - q: float, the interpolated value of the pixel at position (r, c)
    """
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return 0
    r_int = int(floor(r - 0.5))
    c_int = int(floor(c - 0.5))
    q = 0
    p = 0

    for j in range(4):
        c_neighbor = c_int - 1 + j
        p = 0
        if c_neighbor < 0 or c_neighbor >= cols:
            continue

        for i in range(4):
            r_neighbor = r_int - 1 + i
            if r_neighbor < 0 or r_neighbor >= rows:
                continue

            p = p + image[r_neighbor, c_neighbor] * _njit_cubic(r - (r_neighbor + 0.5))
        q = q + p * _njit_cubic(c - (c_neighbor + 0.5))

    return q


def shift_magnify(
    image: np.ndarray,
    shift_row: np.ndarray,
    shift_col: np.ndarray,
    magnification_row: float,
    magnification_col: float,
) -> np.ndarray:
    """
    Shift and magnify using nearest neighbor interpolation.
    :param image: 3D numpy array to interpolate with size (nFrames, nRow, nCol)
    :param shift_row: 1D array with size (nFrames) with values to shift the rows
    :param shift_col: 1D array with size (nFrames) with values to shift the cols
    :param magnification_row: float magnification factor for the rows
    :param magnification_col: float magnification factor for the cols
    :return: 3D float32 numpy array with the result
    """

    nFrames = image.shape[0]
    rows = image.shape[1]
    cols = image.shape[2]
    rowsM = int(rows * magnification_row)
    colsM = int(cols * magnification_col)

    image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
    for f in range(nFrames):
        for j in range(colsM):
            col = j / magnification_col - shift_col[f]
            for i in range(rowsM):
                row = i / magnification_row - shift_row[f]
                image_out[f, i, j] = _interpolate(image[f, :, :], row, col, rows, cols)

    return image_out


@njit(cache=True, parallel=True)
def njit_shift_magnify(
    image: np.ndarray,
    shift_row: np.ndarray,
    shift_col: np.ndarray,
    magnification_row: float,
    magnification_col: float,
) -> np.ndarray:
    """
    Shift and magnify using nearest neighbor interpolation.
    :param image: 3D numpy array to interpolate with size (nFrames, nRow, nCol)
    :param shift_row: 1D array with size (nFrames) with values to shift the rows
    :param shift_col: 1D array with size (nFrames) with values to shift the cols
    :param magnification_row: float magnification factor for the rows
    :param magnification_col: float magnification factor for the cols
    :return: 3D float32 numpy array with the result
    """

    nFrames = image.shape[0]
    rows = image.shape[1]
    cols = image.shape[2]
    rowsM = int(rows * magnification_row)
    colsM = int(cols * magnification_col)

    image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
    for f in range(nFrames):
        for j in prange(colsM):
            col = j / magnification_col - shift_col[f]
            for i in range(rowsM):
                row = i / magnification_row - shift_row[f]
                image_out[f, i, j] = _njit_interpolate(image[f, :, :], row, col, rows, cols)

    return image_out
