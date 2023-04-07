double _c_lanczos_kernel(double v);
float _c_interpolate(__global float *image, float r, float c, int rows, int cols);

#define TAPS 4
#define HALF_TAPS 2

// c2cl-function: _c_lanczos_kernel from _c_interpolation_lanczos.c
double _c_lanczos_kernel(double v) {
  if (v == 0) {
    return 1.0;
  } else if (fabs(v) < TAPS) {
    double v_pi = v * M_PI;
    return TAPS * sin(v_pi) * sin(v_pi / TAPS) / (v_pi * v_pi);
  } else {
    return 0.0;
  }
}

// c2cl-function: _c_interpolate from _c_interpolation_lanczos.c
float _c_interpolate(__global float *image, float r, float c, int rows, int cols) {
  // return 0 if r OR c positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor(r - 0.5);
  const int c_int = (int)floor(c - 0.5);
  double v_interpolated = 0;

  double weight = 0;
  double weight_sum = 0;

  int r_neighbor, c_neighbor;
  double row_factor, col_factor;

  for (int j = 0; j <= TAPS; j++) {
    c_neighbor = c_int - HALF_TAPS + j;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }
    col_factor = _c_lanczos_kernel(c - (c_neighbor + 0.5));

    for (int i = 0; i <= TAPS; i++) {
      r_neighbor = r_int - HALF_TAPS + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }
      row_factor = _c_lanczos_kernel(r - (r_neighbor + 0.5));

      // Add the contribution from this tap to the interpolation
      weight = row_factor * col_factor;
      v_interpolated += image[r_neighbor * cols + c_neighbor] * weight;
      weight_sum += weight;
    }
  }
  return v_interpolated / weight_sum;
}

// tag-copy: _le_interpolation_*.cl
__kernel void
shiftAndMagnify(__global float *image_in, __global float *image_out,
                __global float *shift_row, __global float *shift_col,
                float magnification_row, float magnification_col) {

  int f = get_global_id(0);
  int rM = get_global_id(1);
  int cM = get_global_id(2);

  int rowsM = get_global_size(1);
  int colsM = get_global_size(2);
  int rows = (int)(rowsM / magnification_row);
  int cols = (int)(colsM / magnification_col);
  int nPixels = rowsM * colsM;

  float row = rM / magnification_row - shift_row[f];
  float col = cM / magnification_col - shift_col[f];

  image_out[f * nPixels + rM * colsM + cM] =
      _c_interpolate(&image_in[f * rows * cols], row, col, rows, cols);
}

__kernel void shiftScaleRotate(__global float *image_in,
                               __global float *image_out,
                               __global float *shift_row,
                               __global float *shift_col, float scale_row,
                               float scale_col, float angle) {
  // these are the indexes of the loop
  int f = get_global_id(0);
  int rM = get_global_id(1);
  int cM = get_global_id(2);

  // these are the sizes of the array
  // int nFrames = get_global_size(0);
  int rows = get_global_size(1);
  int cols = get_global_size(2);

  float center_col = cols / 2;
  float center_row = rows / 2;

  float a = cos(angle) / scale_col;
  float b = -sin(angle) / scale_col;
  float c = sin(angle) / scale_row;
  float d = cos(angle) / scale_row;

  int nPixels = rows * cols;

  float col = (a * (cM - center_col - shift_col[f]) +
               b * (rM - center_row - shift_row[f])) +
              center_col;
  float row = (c * (cM - center_col - shift_col[f]) +
               d * (rM - center_row - shift_row[f])) +
              center_row;

  image_out[f * nPixels + rM * cols + cM] =
      _c_interpolate(&image_in[f * nPixels], row, col, rows, cols);
}
// tag-end
