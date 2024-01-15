float _c_interpolate(__global float *image, float r, float c, int rows, int cols);


float _c_interpolate(__global float *image, float r, float c, int rows, int cols) {
  // return 0 if x OR y positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor((float) (r - 0.5));
  const int c_int = (int)floor((float) (c - 0.5));

  double a[4];
  double b[4];
  double dr = r - (r_int + 0.5);
  double dc = c - (c_int + 0.5);
  double dr2 = dr * dr;
  double dr3 = dr2 * dr;
  double dc2 = dc * dc;
  double dc3 = dc2 * dc;

  // Calculate the coefficients for the cubic polynomial
  a[0] = -0.5 * dc3 + dc2 - 0.5 * dc;
  a[1] = 1.5 * dc3 - 2.5 * dc2 + 1;
  a[2] = -1.5 * dc3 + 2 * dc2 + 0.5 * dc;
  a[3] = 0.5 * dc3 - 0.5 * dc2;

  b[0] = -0.5 * dr3 + dr2 - 0.5 * dr;
  b[1] = 1.5 * dr3 - 2.5 * dr2 + 1;
  b[2] = -1.5 * dr3 + 2 * dr2 + 0.5 * dr;
  b[3] = 0.5 * dr3 - 0.5 * dr2;

  double v_interpolated = 0;
  double weight = 0;
  double weight_sum = 0;

  int r_neighbor, c_neighbor;
  double row_factor, col_factor;

  for (int j = 0; j <= 3; j++) {
    c_neighbor = c_int - 1 + j;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }
    col_factor = a[j];

    for (int i = 0; i <= 3; i++) {
      r_neighbor = r_int - 1 + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }
      row_factor = b[i];

      weight = row_factor * col_factor;
      v_interpolated += image[r_neighbor * cols + c_neighbor] * weight;
      weight_sum += weight;
    }
  }
  return v_interpolated / weight_sum;
}


__kernel void
shiftAndMagnify(__global float *image_in, __global float *image_out,
                 float shift_row,  float shift_col,
                float magnification_row, float magnification_col) { 

  int f = get_global_id(0);
  int rM = get_global_id(1);
  int cM = get_global_id(2);

  int rowsM = get_global_size(1);
  int colsM = get_global_size(2);
  int rows = (int)(rowsM / magnification_row);
  int cols = (int)(colsM / magnification_col);
  int nPixels = rowsM * colsM;

  float row = rM / magnification_row - shift_row;
  float col = cM / magnification_col - shift_col;

  image_out[f * nPixels + rM * colsM + cM] =
      _c_interpolate(&image_in[f * rows * cols], row, col, rows, cols);
}

__kernel void shiftScaleRotate(__global float *image_in,
                               __global float *image_out,
                              float shift_row,
                              float shift_col, float scale_row,
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

  float col = (a * (cM - center_col - shift_col) +
               b * (rM - center_row - shift_row)) +
              center_col;
  float row = (c * (cM - center_col - shift_col) +
               d * (rM - center_row - shift_row)) +
              center_row;

  image_out[f * nPixels + rM * cols + cM] =
      _c_interpolate(&image_in[f * nPixels], row, col, rows, cols);
}

__kernel void PolarTransform(__global float *image_in,
                               __global float *image_out, 
                               int og_row, int og_col, int scale) {
                                
  // these are the indexes of the loop we are in
  int f = get_global_id(0);
  int rM = get_global_id(1);
  int cM = get_global_id(2);

  // these are the sizes of the output array
  // int nFrames = get_global_size(0);
  int rows = get_global_size(1);
  int cols = get_global_size(2);

  float center_col = og_col / 2;
  float center_row = og_row / 2;

  float max_radius = hypot(center_row, center_col);
  float pi = 4 * atan((float)(1.0));
  
  float angle =  rM * 2 * pi  / (rows-1);

  float radius;
  if (scale==1) {
    radius = exp(cM*log(max_radius)/(cols-1));
  } else {
    radius = cM * max_radius / (cols-1);
  }

  float col = radius * cos(angle) + center_col;
  float row = radius * sin(angle) + center_row;

  image_out[f * rows * cols + rM * cols + cM] =
      _c_interpolate(&image_in[f * og_col*og_row], row, col, og_row, og_col);
}

