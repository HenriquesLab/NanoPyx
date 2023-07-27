__kernel void
conv2d(__global float *image_in, __global float *image_out, __global float *kernel_array,
      int nRows_kernel, int nCols_kernel, int center_r, int center_c) {

  int r = get_global_id(0);
  int c = get_global_id(1);

  int nRows = get_global_size(0);
  int nCols = get_global_size(1);

  float acc = 0;

  int local_row, local_col;

  for (int kr = 0; kr<nRows_kernel; kr++) {
    for (int kc = 0; kc<nCols_kernel; kc++) {
      local_row = min(max(r+(kr-center_r),0),nRows-1);
      local_col = min(max(c+(kc-center_c),0),nCols-1);
      acc = acc+ kernel_array[kr*nCols_kernel+kc] * image_in[local_row*nCols+local_col];
      }
    }

  image_out[r*nCols+c] = acc;
}

