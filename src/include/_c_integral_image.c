
void _c_integral_image(float* padded,float* integral,int n_row,int n_col,int t_row,int t_col,float var_diff) {
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
    int row, col;
    int row_start = (int)(1 > -t_row ? 1 : -t_row);
    int row_end = (int)(n_row < n_row - t_row ? n_row : n_row - t_row);
    float t, distance;

    for (row = row_start; row < row_end; ++row) {
        for (col = 1; col < n_col - t_col; ++col) {
            distance = 0;
            t = (padded[row * n_col + col] -
                    padded[(row + t_row) * n_col + (col + t_col)]);
            distance += t * t;
            distance -= var_diff;
            integral[row * n_col + col] = (distance +
                                           integral[(row - 1) * n_col + col] +
                                           integral[row * n_col + col - 1] -
                                           integral[(row - 1) * n_col + col - 1]);
        }
    }
}