

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


__kernel void nlm_denoising(__global float *padded,__global int *shifts, __global float *weights, __global float *integral,__global float *result,int n_row, int n_col, int patch_size, int patch_distance, float h2s2, float var) {

    int f = get_global_id(0);
    int nframes = get_global_size(0);
    int shift = get_global_id(1);

    int t_row = shifts[shift*2];
    int t_col = shifts[shift*2+1];

    int offset = patch_size / 2;
    int row_start = max(offset, offset-t_row);
    int row_end = min(n_row-offset, n_row-offset-t_row);

    float alpha = 1;
    if(t_col==0){
        alpha = 0.5;
    }

    _c_integral_image(&padded[f*n_row*n_col],&integral[shift*nframes*n_row*n_col+f*n_row*n_col],n_row,n_col,t_row,t_col,var);

    int row,col;
    int row_shift, col_shift;
    float distance, weight;
    for(row=row_start;row<row_end;row++){
        for(col=offset;col<n_col-offset-t_col;col++){
            row_shift = row+t_row;
            col_shift = col+t_col;
            distance = _c_integral_to_distance(&integral[shift*nframes*n_row*n_col+f*n_row*n_col],n_row,n_col,row,col,offset,h2s2);
            weight = alpha * exp(-distance);

            weights[shift*nframes*n_row*n_col+f*n_row*n_col+row*n_col+col] = weights[shift*nframes*n_row*n_col+f*n_row*n_col+row*n_col+col] + weight;
            weights[shift*nframes*n_row*n_col+f*n_row*n_col+row_shift*n_col+col_shift] = weights[shift*nframes*n_row*n_col+f*n_row*n_col+row_shift*n_col+col_shift] + weight;

            result[shift*nframes*n_row*n_col+f*n_row*n_col+row*n_col+col] = result[shift*nframes*n_row*n_col+f*n_row*n_col+row*n_col+col] + weight * padded[f*n_row*n_col+row_shift*n_col+col_shift];
            result[shift*nframes*n_row*n_col+f*n_row*n_col+row_shift*n_col+col_shift] = result[shift*nframes*n_row*n_col+f*n_row*n_col+row_shift*n_col+col_shift] + weight * padded[f*n_row*n_col+row*n_col+col];
        }
    }
}

__kernel void nlm_normalizer(__global float *weights, __global float *result, __global float *output, int pad_size, int n_shifts, int n_row, int n_col){

    int f = get_global_id(0);
    int nframes = get_global_size(0);
    int f_row = get_global_id(1) + pad_size;
    int f_col = get_global_id(2) + pad_size;

    float final_result = 0;
    float final_weight = 0;
    
    int shift;
    for(shift=0;shift<n_shifts;shift++){
        final_result = final_result + result[shift*nframes*n_row*n_col+f*n_row*n_col+f_row*n_col+f_col];
        final_weight = final_weight + weights[shift*nframes*n_row*n_col+f*n_row*n_col+f_row*n_col+f_col];
    }

    output[f*n_row*n_col+f_row*n_col+f_col] = final_result / final_weight;
    
}