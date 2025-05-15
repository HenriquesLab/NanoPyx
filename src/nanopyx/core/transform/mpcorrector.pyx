# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from cython.parallel import prange
import numpy as np
cimport numpy as np

def macro_pixel_corrector(img: np.ndarray, magnification: int) -> np.ndarray:
    """
    Corrects the macro-pixel effect.

    Parameters
    ----------
    img : np.ndarray
        The input image to be corrected.
    magnification : int
        The magnification factor used.

    Returns
    -------
    np.ndarray
        The corrected image.
    """
    
    return np.asarray(_macro_pixel_corrector(img.astype(np.float32), magnification), dtype=np.float32)

cdef float[:, :, :] _macro_pixel_corrector(float[:, :, :] img, int magnification):
    """
    Cython implementation of the macro-pixel correction.

    Parameters
    ----------
    img : float[:, :, :]
        The input image to be corrected.
    magnification : int
        The magnification factor used.

    Returns
    -------
    float[:, :, :]
        The corrected image.
    """
    cdef int slices = img.shape[0]
    cdef int rowsM = img.shape[1]
    cdef int colsM = img.shape[2]
    cdef int rows = rowsM // magnification
    cdef int cols = colsM // magnification

    cdef float map_value
    cdef int x, y, offset, s, r, c

    cdef float[:, :] map

    for s in range(slices):
        map = np.zeros((magnification, magnification), dtype=np.float32)
        for ry in range(rows):
            for rx in range(cols):
                for y in range(magnification):
                    for x in range(magnification):
                        # global coordinates in original image
                        gx = rx * magnification + x
                        gy = ry * magnification + y
                        map[y, x] += img[s, gy, gx]
        map = np.asarray(map) / rows*cols

        for yM in range(rowsM):
            for xM in range(colsM):
                y = yM // magnification
                x = xM // magnification
                y_offset = yM - y * magnification
                x_offset = xM - x * magnification
                correction = map[y_offset, x_offset]
                if correction != 0:
                    img[s, yM, xM] /= correction

    return img
                