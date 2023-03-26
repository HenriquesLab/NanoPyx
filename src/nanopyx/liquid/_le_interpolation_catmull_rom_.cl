double _c_interpolate(__global float *image, double r, double c, int rows, int cols);
double _c_cubic(double v);

// c2cl-function: _c_cubic from _c_interpolation_catmull_rom.c
double _c_cubic(double v) {
  double a = 0.5;
  double z = 0;
  if (v < 0) {
    v = -v;
  }
  if (v < 1) {
    z = v * v * (v * (-a + 2) + (a - 3)) + 1;
  } else if (v < 2) {
    z = -a * v * v * v + 5 * a * v * v - 8 * a * v + 4 * a;
  }
  return z;
}

// c2cl-function: _c_interpolate from _c_interpolation_catmull_rom.c
double _c_interpolate(__global float *image, double r, double c, int rows, int cols) {
  // return 0 if x OR y positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r0 = (int)r;
  const int c0 = (int)c;

  const int u0 = (int)floor(r - 0.5);
  const int v0 = (int)floor(c - 0.5);
  double q = 0;
  double p = 0;

  int u, v;

  for (int j = 0; j < 4; j++) {
    v = v0 - 1 + j;
    p = 0;
    for (int i = 0; i < 4; i++) {
      u = u0 - 1 + i;
      int _u = (int)fmax(0, fmin(u, rows - 1));
      int _v = (int)fmax(0, fmin(v, cols - 1));
      p = p + image[_u * cols + _v] * _c_cubic(r - (u + 0.5));
    }
    q = q + p * _c_cubic(c - (v + 0.5));
  }
  return q;
}

__kernel void
shiftAndMagnify(__global float *image_in, __global float *image_out,
                __global float *shift_row, __global float *shift_col,
                float magnification_row, float magnification_col) {

  int f = get_global_id(0);
  int rM = get_global_id(1);
  int cM = get_global_id(2);

  // int nFrames = get_global_size(0);
  int rowsM = get_global_size(1);
  int colsM = get_global_size(2);
  int rows = (int)(rowsM / magnification_row);
  int cols = (int)(colsM / magnification_col);
  int nPixels = rowsM * colsM;

  float row = rM / magnification_row - shift_row[f];
  float col = cM / magnification_col - shift_col[f];

  image_out[f * nPixels + rM * colsM + cM] =
      _c_interpolate(&image_in[f * nPixels], row, col, rows, cols);
}

__kernel void ShiftScaleRotate(__global float *image_in,
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

  float center_rowM = (rows * scale_row) / 2;
  float center_colM = (cols * scale_col) / 2;

  float a = cos(angle) / scale_col;
  float b = -sin(angle);
  float c = sin(angle);
  float d = cos(angle) / scale_row;

  int nPixels = rows * cols;

  float col = (a * (cM - center_colM) + b * (rM - center_rowM)) - shift_col[f] +
              center_col;
  float row = (c * (cM - center_colM) + d * (rM - center_rowM)) - shift_row[f] +
              center_row;

  image_out[f * nPixels + rM * cols + cM] =
      _c_interpolate(&image_in[f * nPixels], row, col, rows, cols);
}
