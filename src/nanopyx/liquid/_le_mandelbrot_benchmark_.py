import warnings

import numpy as np

try:
    from numba import njit, prange

except ImportError:
    # raise a warning that numba is not installed
    # and that the njit functions will not be used
    # and that the pure python functions will be used instead

    prange = range

    def njit(*args, **kwargs):
        def wrapper(func):
            warnings.warn(
                f"Numba is not installed. Using pure python for {func.__name__}"
            )
            return func

        return wrapper


MAX_ITERATIONS = 1000
DIVERGENCE = 10


def _py_mandelbrot(row: float, col: float) -> int:
    zrow = 0
    zcol = 0
    iterations = 0
    while zrow * zrow + zcol * zcol <= DIVERGENCE and iterations < MAX_ITERATIONS:
        zrow_new = zrow * zrow - zcol * zcol + row
        zcol_new = 2 * zrow * zcol + col
        zrow = zrow_new
        zcol = zcol_new
        iterations += 1

    return iterations


@njit(cache=True)
def _njit_mandelbrot(row: float, col: float) -> int:
    zrow = 0
    zcol = 0
    iterations = 0
    while zrow * zrow + zcol * zcol <= DIVERGENCE and iterations < MAX_ITERATIONS:
        zrow_new = zrow * zrow - zcol * zcol + row
        zcol_new = 2 * zrow * zcol + col
        zrow = zrow_new
        zcol = zcol_new
        iterations += 1

    return iterations


def mandelbrot(
    image: np.ndarray,
    r_start: float,
    r_end: float,
    c_start: float,
    c_end: float,
) -> np.ndarray:
    """
    Mandelbrot set generator.
    :param image: numpy array to store the result
    :param r_start: start of the real axis
    :param r_end: end of the real axis
    :param c_start: start of the complex axis
    :param c_end: end of the complex axis
    :return: numpy array with the result
    """
    rows, cols = image.shape
    for row in range(rows):
        for col in range(cols):
            image[row, col] = _py_mandelbrot(
                r_start + (r_end - r_start) * row / rows,
                c_start + (c_end - c_start) * col / cols,
            )
    return image


@njit(cache=True, parallel=True)
def njit_mandelbrot(
    image: np.ndarray,
    r_start: float,
    r_end: float,
    c_start: float,
    c_end: float,
) -> np.ndarray:
    """
    Mandelbrot set generator.
    :param image: numpy array to store the result
    :param r_start: start of the real axis
    :param r_end: end of the real axis
    :param c_start: start of the complex axis
    :param c_end: end of the complex axis
    :return: numpy array with the result
    """
    rows, cols = image.shape
    for row in prange(rows):
        for col in range(cols):
            image[row, col] = _njit_mandelbrot(
                r_start + (r_end - r_start) * row / rows,
                c_start + (c_end - c_start) * col / cols,
            )
    return image
