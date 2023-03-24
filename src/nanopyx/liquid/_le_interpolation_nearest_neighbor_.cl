float _c_interpolate(__global float *image, float row, float col, int rows,
                     int cols);

// c2cl-function: _c_interpolate from _c_interpolation_nearest_neighbor.c
float _c_interpolate(__global float *image, float row, float col, int rows,
                     int cols) {
  int r = (int)row;
  int c = (int)col;
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }
  return image[r * cols + c];
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

__kernel void
ShiftScaleRotate(__global float *image_in, __global float *image_out,
                __global float *shift_row, __global float *shift_col,
                float scale_row, float scale_col, float angle) {
  // these are the indexes of the loop                
  int f = get_global_id(0);
  int rM = get_global_id(1);
  int cM = get_global_id(2);

  // these are the sizes of the array
  // int nFrames = get_global_size(0);
  int rows = get_global_size(1);
  int cols = get_global_size(2);

  float center_col = cols/2;
  float center_row = rows/2;

  float center_rowM = (rows * scale_row) / 2;
  float center_colM = (cols * scale_col) / 2;
  
  float a = cos(angle)/scale_col;
  float b = -sin(angle);
  float c = sin(angle);
  float d = cos(angle)/scale_row;
  
  int nPixels = rows * cols;

  float col = (a*(cM-center_colM)+b*(rM-center_rowM)) - shift_col[f] + center_col;
  float row = (c*(cM-center_colM)+d*(rM-center_rowM)) - shift_row[f] + center_row;

  image_out[f * nPixels + rM * cols + cM] =
      _c_interpolate(&image_in[f * nPixels], row, col, rows, cols);
}