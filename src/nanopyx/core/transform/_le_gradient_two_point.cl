void _c_gradient_two_point(__global float* image, __global float* imGc, __global float* imGr, int rows, int cols);

void _c_gradient_two_point(__global float* image, __global float* imGc, __global float* imGr, int rows, int cols) {
    int c1, r1;
    int c0, r0;
    int offset, left_offset, top_offset;

    for (r1 = 0; r1 < rows; r1++) {
        for (c1 = 0; c1 < cols; c1++) {

            // compute previous column/row with clamp at 0 (max(x-1,0))
            c0 = c1 > 0 ? c1 - 1 : 0;
            r0 = r1 > 0 ? r1 - 1 : 0;

            offset = r1 * cols + c1;
            left_offset = r1 * cols + c0;  // (x0, y1)
            top_offset  = r0 * cols + c1;  // (x1, y0)

            // 2-point gradient (same as calculateGradient_2point)
            imGc[offset] = image[offset] - image[left_offset];
            imGr[offset] = image[offset] - image[top_offset];
        }
    }
}

__kernel void gradient_two_point(__global float* image,
                                     __global float* imGc,
                                     __global float* imGr,
                                     int rows,
                                     int cols)
        {
            int f = get_global_id(0);

            _c_gradient_two_point(&image[f*rows*cols], &imGc[f*rows*cols], &imGr[f*rows*cols], rows, cols);
        }