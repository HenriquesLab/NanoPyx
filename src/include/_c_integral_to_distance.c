float _c_integral_to_distance(const float* integral,int rows,int cols,int row,int col,int offset,float h2s2) {
    /*
    Parameters
    ----------
    integral : PyListObject*
        The integral image as computed by `_integral_image_2d`.
    rows, cols : int
        Number of rows and columns in the integral image.
    row, col : int
        Index of the patch's center pixel.
    offset : int
        The non-local means patch radius.
    h2s2 : float
        Normalization factor related to the image standard deviation and `h`
        parameter.

    Returns
    -------
    distance : float
        The patch distance

    Notes
    -----
    Used in `_fast_nl_means_denoising_2d` which is a fast non-local means
    algorithm using integral images as described in [1]_, [2]_.

    References
    ----------
    .. [1] J. Darbon, A. Cunha, T.F. Chan, S. Osher, and G.J. Jensen. Fast
        nonlocal filtering applied to electron cryomicroscopy, in 5th IEEE
        International Symposium on Biomedical Imaging: From Nano to Macro,
        2008, pp. 1331-1334.
    .. [2] Jacques Froment. Parameter-Free Fast Pixelwise Non-Local Means
        Denoising. Image Processing On Line, 2014, vol. 4, pp. 300-326.
    */
    int row_plus_offset = (int)(row + offset);
    int row_minus_offset = (int)(row - offset);
    int col_plus_offset = (int)(col + offset);
    int col_minus_offset = (int)(col - offset);

    float distance = (integral[row_plus_offset * cols + col_plus_offset] +
                      integral[row_minus_offset * cols + col_minus_offset] -
                      integral[row_minus_offset * cols + col_plus_offset] -
                      integral[row_plus_offset * cols + col_minus_offset]);

    return (distance > 0.0f) ? distance / h2s2 : 0.0f;
}
