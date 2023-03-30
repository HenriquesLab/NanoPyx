import numpy as np

from .__njit__ import njit, prange


def _interpolate(image, row, col, rows, cols):
    r = int(row)
    c = int(col)
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return 0
    else:
        return image[r, c]


@njit(cache=True)
def _njit_interpolate(image, row, col, rows, cols):
    r = int(row)
    c = int(col)
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return 0
    else:
        return image[r, c]


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
                image_out[f, i, j] = _njit_interpolate(
                    image[f, :, :], row, col, rows, cols
                )

    return image_out


def shift_scale_rotate(
    image: np.ndarray,
    shift_row: np.ndarray,
    shift_col: np.ndarray,
    scale_row: float,
    scale_col: float,
    angle: float,
) -> np.ndarray:
    """
    Shift, magnify and rotate using nearest neighbor interpolation.
    The order of operations is SCALE AND ROTATE AROUND CENTER THEN SHIFT
    :param image: 3D numpy array to interpolate with size (nFrames, nRow, nCol)
    :param shift_row: 1D array with size (nFrames) with values to shift the rows
    :param shift_col: 1D array with size (nFrames) with values to shift the cols
    :param scale_row: float scale factor for the rows
    :param scale_col: float scale factor for the cols
    :param angle: float angle of rotation in radians. positive is counter clockwise
    :return: 3D float32 numpy array with the result
    """

    nFrames = image.shape[0]
    rows = image.shape[1]
    cols = image.shape[2]

    center_row = rows / 2
    center_col = cols / 2
    # center_rowM = (rows * scale_row) / 2
    # center_colM = (cols * scale_col) / 2

    # Composing an affine transform
    # Its scale => rotate => shift, but we iterate the final image so shift is the first operation on the vector
    # SCALE     ROTATE         SHIFT
    # sx  0 0   +cos -sin 0    0 0 tx   j     col
    #  0 sy 0 . +sin +cos 0  . 0 0 ty . i  =  row
    #  0  0 1     0    0  1    0 0  1   1      1

    # After calculations we have
    # SHIFT . SCALE . ROTATE = a  b  tcol
    #                          c  d  trow
    #                          0  0   1
    # We multiply the matrix by every vector (i,j,1)
    
    a = np.cos(angle) / scale_col
    b = -np.sin(angle)/ scale_col
    c = np.sin(angle) / scale_row
    d = np.cos(angle) / scale_row
    
    # Note#1:tcol and trow are simply shift_col and shift_row rotated and thus are functions of a,b,c,d
    #   In the below code we simplify it by separating it by their common factors a,b,c,d

    # Note#2: In reality we have to translate by the center before and after to have centered coordinates
    # In order to keep the same image size during scaling the translation for centered coordinates is given by 
    # (center_magnified - center_og) - center_magnified == center_og
    # This can be seen by noting that when (i,j)=(0,0) we are actually at (center_magnified - center_og) coordinates on the scaled image

    image_out = np.zeros((nFrames, rows, cols), dtype=np.float32)
    for f in range(nFrames):
        for j in range(cols):
            for i in range(rows):
                col = (
                    (a * (j - center_col-shift_col[f]) + b * (i - center_row-shift_row[f]))
                    + center_col
                )
                row = (
                    (c * (j - center_col-shift_col[f]) + d * (i - center_row-shift_row[f]))
                    + center_row
                )
                image_out[f, i, j] = _interpolate(image[f, :, :], row, col, rows, cols)

    return image_out


@njit(cache=True, parallel=True)
def njit_shift_scale_rotate(
    image: np.ndarray,
    shift_row: np.ndarray,
    shift_col: np.ndarray,
    scale_row: float,
    scale_col: float,
    angle: float,
) -> np.ndarray:
    """
    Shift, magnify and rotate using nearest neighbor interpolation.
    The order of operations is SCALE AND ROTATE AROUND CENTER THEN SHIFT
    :param image: 3D numpy array to interpolate with size (nFrames, nRow, nCol)
    :param shift_row: 1D array with size (nFrames) with values to shift the rows
    :param shift_col: 1D array with size (nFrames) with values to shift the cols
    :param scale_row: float scale factor for the rows
    :param scale_col: float scale factor for the cols
    :param angle: float angle of rotation in radians. positive is counter clockwise
    :return: 3D float32 numpy array with the result
    """

    nFrames = image.shape[0]
    rows = image.shape[1]
    cols = image.shape[2]

    center_row = rows / 2
    center_col = cols / 2

    # center_rowM = (rows * scale_row) / 2
    # center_colM = (cols * scale_col) / 2

    a = np.cos(angle) / scale_col
    b = -np.sin(angle) / scale_col
    c = np.sin(angle) / scale_row
    d = np.cos(angle) / scale_row

    image_out = np.zeros((nFrames, rows, cols), dtype=np.float32)
    for f in range(nFrames):
        for j in prange(cols):
            for i in range(rows):
                col = (
                    (a * (j - center_col-shift_col[f]) + b * (i - center_row-shift_row[f]))
                    + center_col
                )
                row = (
                    (c * (j - center_col-shift_col[f]) + d * (i - center_row-shift_row[f]))
                    + center_row
                )
                image_out[f, i, j] = _njit_interpolate(image[f, :, :], row, col, rows, cols)

    return image_out