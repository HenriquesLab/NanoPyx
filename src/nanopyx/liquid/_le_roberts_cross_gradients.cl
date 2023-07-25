void _c_gradient_roberts_cross(__global float* image, __global float* imGc, __global float* imGr, int rows, int cols);

void _c_gradient_roberts_cross(__global float* image, __global float* imGc, __global float* imGr, int rows, int cols) {
    int c1, r1, c0, r0;
    float im_c0_r1, im_c1_r0, im_c0_r0, im_c1_r1;

    for (r1 = 0; r1 < rows; r1++) {
        for (c1 = 0; c1 < cols; c1++) {

            c0 = c1 > 0 ? c1 - 1 : 0;
            r0 = r1 > 0 ? r1 - 1 : 0;

            im_c0_r1 = image[r0 * cols + c1];
            im_c1_r0 = image[r1 * cols + c0];
            im_c0_r0 = image[r0 * cols + c0];
            im_c1_r1 = image[r1 * cols + c1];

            imGc[r1 * cols + c1] = im_c0_r1 - im_c1_r0 + im_c1_r1 - im_c0_r0;
            imGr[r1 * cols + c1] = -im_c0_r1 + im_c1_r0 + im_c1_r1 - im_c0_r0;
        }
    }
}

__kernel void gradient_roberts_cross(__global float* image,
                                     __global float* imGc,
                                     __global float* imGr,
                                     int rows,
                                     int cols)
        {
            int f = get_global_id(0);

            _c_gradient_roberts_cross(&image[f*rows*cols], &imGc[f*rows*cols], &imGr[f*rows*cols], rows, cols);
        }