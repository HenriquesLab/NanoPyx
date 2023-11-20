#include <Python.h>
#include <stddef.h>

void _c_integral_image(
    float* padded,
    float* integral,
    int n_row,
    int n_col,
    int n_channels,
    int t_row,
    int t_col,
    float var_diff) {
    /*
    Parameters
    ----------
    padded : PyListObject*
        Image of interest.
    integral : PyListObject*
        Output of the function. The list is filled with integral values.
        `integral` should have the same shape as `padded`.
    t_row : int
        Shift along the row axis.
    t_col : int
        Shift along the column axis (positive).
    n_row : int
    n_col : int
    n_channels : int
    var_diff : float
        The double of the expected noise variance.  If non-zero, this
        is used to reduce the apparent patch distances by the expected
        distance due to the noise.

    Notes
    -----

    The integral computation could be performed using
    `transform.integral_image`, but this helper function saves memory
    by avoiding copies of `padded`.
    */
    int row, col, channel;
    int row_start = (int)(1 > -t_row ? 1 : -t_row);
    int row_end = (int)(n_row < n_row - t_row ? n_row : n_row - t_row);
    float t, distance;

    for (row = row_start; row < row_end; ++row) {
        for (col = 1; col < n_col - t_col; ++col) {
            distance = 0;
            for (channel = 0; channel < n_channels; ++channel) {
                t = (float)(PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(padded, row), col * n_channels + channel)) -
                            PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(padded, row + t_row), (col + t_col) * n_channels + channel)));
                distance += t * t;
            }
            distance -= n_channels * var_diff;
            PyList_SetItem(PyList_GetItem(integral, row), col, Py_BuildValue(
                "f", distance +
                (float)PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(integral, row - 1), col)) +
                (float)PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(integral, row), col - 1)) -
                (float)PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(integral, row - 1), col - 1))));
        }
    }
}