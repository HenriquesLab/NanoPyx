__kernel void pearsons_correlation(
    __global const float *im1,
    __global const float *im2,
    __global float *sum11,
    __global float *sum12,
    __global float *sum22,
    float mean_1,
    float mean_2)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    int n_row = get_global_size(0);

    float d_im1 = im1[row*n_row + col] - mean_1;
    float d_im2 = im2[row*n_row + col] - mean_2;

    sum11[row*n_row + col] = d_im1 * d_im1;
    sum12[row*n_row + col] = d_im1  * d_im2;
    sum22[row*n_row + col] = d_im2 * d_im2;

}