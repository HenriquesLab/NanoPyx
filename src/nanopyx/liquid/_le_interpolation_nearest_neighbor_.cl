float _c_interpolate(__global float *image, float row, float col, int rows, int cols);

// c2cl-function: _c_interpolate from _c_interpolation_nearest_neighbor.c
float _c_interpolate(__global float *image, float row, float col, int rows, int cols) {
  int r = (int)row;
  int c = (int)col;
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }
  return image[r * cols + c];
}

// tag-start: _le_interpolation_*.cl
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

// // tag-start: _le_interpolation_nearest_neighbor_.cl.PolarTransform
// __kernel void PolarTransform(__global float *image_in,
//                                __global float *image_out, 
//                                int og_row, int og_col, int scale) {
                                
//   // these are the indexes of the loop
//   int f = get_global_id(0);
//   int rM = get_global_id(1);
//   int cM = get_global_id(2);

//   // these are the sizes of the array
//   // int nFrames = get_global_size(0);
//   int rows = get_global_size(1);
//   int cols = get_global_size(2);

//   float center_col = og_col / 2;
//   float center_row = og_row / 2;

//   float max_radius = hypot(center_col, center_row);

//   float pi = 4 * atan(1);

//   float angle =  rM * 2 * pi  / (rows-1);
//   float radius;
//   if (scale==1) {
//     radius = exp(cM*log(max_radius)/(cols-1));
//   } else {
//     radius = cM * max_radius / (cols-1);
//   }

//   float col = radius * cos(angle) + center_col;
//   float row = radius * sin(angle) + center_row;

//   image_out[f * rows * cols + rM * cols + cM] =
//       _c_interpolate(&image_in[f * og_col*og_row], row, col, rows, cols);
// }
// // tag-end
