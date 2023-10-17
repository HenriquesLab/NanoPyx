float _c_calculate_radiality_per_subpixel(int i, int j, __global float* imGx, __global float* imGy, __global float* xRingCoordinates, __global float* yRingCoordinates, int magnification, float ringRadius, int nRingCoordinates, int radialityPositivityConstraint, int h, int w);
float _c_calculate_dk(float x, float y, float xc, float yc, float vGx, float vGy, float GMag, float ringRadius);
double _c_cubic(double v);
float _c_interpolate(__global float* image, float r, float c, int rows, int cols);

// Cubic function used in Catmull-Rom interpolation
// https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull.E2.80.93Rom_spline
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

// Catmull-Rom interpolation
float _c_interpolate(__global float* image, float r, float c, int rows, int cols) {
  // return 0 if r OR c positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor((float)(r - 0.5));
  const int c_int = (int)floor((float)(c - 0.5));
  double q = 0;
  double p = 0;

  int r_neighbor, c_neighbor;

  for (int j = 0; j < 4; j++) {
    c_neighbor = c_int - 1 + j;
    p = 0;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }

    for (int i = 0; i < 4; i++) {
      r_neighbor = r_int - 1 + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }
      p = p + image[r_neighbor * cols + c_neighbor] *
                  _c_cubic(r - (r_neighbor + 0.5));
    }
    q = q + p * _c_cubic(c - (c_neighbor + 0.5));
  }
  return q;
}


float _c_calculate_dk(float x, float y, float xc, float yc, float vGx, float vGy, float GMag, float ringRadius){
    float Dk = 0;
    if (GMag != 0) {
        Dk = 1 - (fabs(vGy * (xc - x) - vGx * (yc - y)) / GMag) / ringRadius;
        Dk = Dk * Dk;
    } 
    return Dk;
}

float _c_calculate_radiality_per_subpixel(int i, int j, __global float* imGx, __global float* imGy, __global float* xRingCoordinates, __global float* yRingCoordinates, int magnification, float ringRadius, int nRingCoordinates, int radialityPositivityConstraint, int h, int w) {
    int sampleIter;
    float x0, y0, xc, yc, xRing, yRing, vGx, vGy, GMag, Dk, DivDFactor = 0, CGH = 0;

    xc = i + 0.5;
    yc = j + 0.5;
    
    for (sampleIter = 0; sampleIter < nRingCoordinates; sampleIter++) {
        xRing = xRingCoordinates[sampleIter];
        yRing = yRingCoordinates[sampleIter];

        x0 = xc + xRing;
        y0 = yc + yRing;

        vGx = _c_interpolate(imGx, y0 / magnification, x0 / magnification, h, w);
        vGy = _c_interpolate(imGy, y0 / magnification, x0 / magnification, h, w);
        GMag = sqrt(vGx * vGx + vGy * vGy);

        Dk = _c_calculate_dk(x0, y0, xc, yc, vGx, vGy, GMag, ringRadius);

        if ((vGx * xRing + vGy * yRing) > 0) {
            DivDFactor -= Dk;
        } else {
            DivDFactor += Dk;
        }
    }

    DivDFactor /= nRingCoordinates;

    if (radialityPositivityConstraint == 1) {
        CGH = fmax(DivDFactor, 0);
    } else {
        CGH = DivDFactor;
    }

    return CGH;
}


__kernel void radiality(__global float *image_in, __global float *imageinterp_in, __global float *gradient_X, __global float *gradient_Y, 
          __global float *image_out, __global float *xRingCoordinates, __global float *yRingCoordinates, int magnification,
          float ringRadius, int nRingCoordinates, int radialityPositivityConstraint, int border, int h, int w) {

    // Indices of the current pixel
    int f = get_global_id(0);
    int row = get_global_id(1);
    int col = get_global_id(2);

    // Total number of rows and columns in the output image
    int nRows = h * magnification;
    int nCols = w * magnification;
    
    // Pixels in the input image
    int nPixels_in = h * w;

    // Pixels in the output image
    int nPixels_out = nRows * nCols;

    // Add border and magnification since we dont start at index zero
    row = (1 + border) * magnification + row;
    col = (1 + border) * magnification + col;

    image_out[f * nPixels_out + row * nCols + col] = _c_calculate_radiality_per_subpixel(col, row, &gradient_X[f*nPixels_in], &gradient_Y[f*nPixels_in], xRingCoordinates, yRingCoordinates, magnification, ringRadius, nRingCoordinates, radialityPositivityConstraint, h, w) 
    * imageinterp_in[f * nPixels_out + row * nCols + col];

}

         