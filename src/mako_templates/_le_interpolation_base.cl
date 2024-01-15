% for h in self.attr.headers:
${h}
% endfor

% for d in self.attr.defines:
#define ${d[0]} ${d[1]}
% endfor

% for f in self.attr.functions:
${f}

% endfor

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

