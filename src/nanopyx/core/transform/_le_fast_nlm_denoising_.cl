

void _c_integral_image(__global float *padded,__global float *integral,int n_row,int n_col,int t_row,int t_col,float var_diff);
float _c_integral_to_distance(const __global float *integral,int rows,int cols,int row,int col,int offset,float h2s2);


void _c_integral_image(__global float *padded,__global float *integral,int n_row,int n_col,int t_row,int t_col,float var_diff) {
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

float _c_integral_to_distance(const __global float *integral,int rows,int cols,int row,int col,int offset,float h2s2) {
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



__kernel void
nlm_denoising(__global float *padded, __global float *result, __global float* integral, __global float* Z, const int f, const int n_row, const int n_col, const int offset, const float var, const float h2s2, const int patch_distance) {
        int t_row = get_global_id(0)-patch_distance;
        int t_col = get_global_id(1);

        int size_col = get_global_size(1);
        int current_patch = get_global_id(0)*size_col*n_row*n_col+t_col*n_row*n_col;

        float  alpha = 1;
        if (t_col == 0){
            alpha = 0.5;
        }
    
        _c_integral_image(&padded[f*n_col*n_row], &integral[current_patch], n_row, n_col, t_row, t_col, var);
        
        int row_start = max(offset,offset-t_row);
        int row_end = min(n_row-offset, n_row-offset-t_row);
        
        int col_start = offset;
        int col_end = n_col-offset-t_col;

        int row_shift, col_shift;

        int row, col;
        float distance, weight;
        for (row=row_start;row<row_end;row++){
            row_shift = row+t_row;
            for (col=col_start;col<col_end;col++){
                distance = _c_integral_to_distance(&integral[current_patch], n_row, n_col, row, col, offset, h2s2);
                if (distance <= 5.0){
                    col_shift = col+t_col;
                    weight = alpha * exp(-distance);
                    Z[f*n_col*n_row+row*n_col+col] = Z[f*n_col*n_row+row*n_col+col] + weight;
                    Z[f*n_col*n_row+row_shift*n_col+col_shift]  = Z[f*n_col*n_row+row_shift*n_col+col_shift] + weight;
                    
                    result[f*n_col*n_row+row*n_col+col] = result[f*n_col*n_row+row*n_col+col] + weight * padded[f*n_col*n_row+row_shift*n_col+col_shift];
                    result[f*n_col*n_row+row_shift*n_col+col_shift]  = result[f*n_col*n_row+row_shift*n_col+col_shift] +  weight * padded[f*n_col*n_row+row*n_col+col];
                }
            }
        }
}

__kernel void
nlm_normalizer(__global float *result, __global float* Z, const int f, const int pad_size) {

        int row = get_global_id(0)+pad_size;
        int col = get_global_id(1)+pad_size;

        int n_row = get_global_size(0)+pad_size*2;
        int n_col = get_global_size(1)+pad_size*2;

        result[f*n_row*n_col + row * n_col + col] = result[f*n_row*n_col + row * n_col + col] / Z[f*n_row*n_col + row * n_col + col];
        
}
