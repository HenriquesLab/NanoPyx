float _c_calculate_rgc(int xM, int yM, __global float* imIntGx, __global float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity);
double _c_calculate_dw(double distance, double tSS);
double _c_calculate_dk(float Gx, float Gy, float dx, float dy, float distance);

// RGC takes as input the interpolated intensity gradients in the x and y directions

// calculate distance weight
double _c_calculate_dw(double distance, double tSS) {
  return pow((distance * exp((-distance * distance) / tSS)), 4);
}

// calculate degree of convergence
double _c_calculate_dk(float Gx, float Gy, float dx, float dy, float distance) {
  float Dk = fabs(Gy * dx - Gx * dy) / sqrt(Gx * Gx + Gy * Gy);
  if (isnan(Dk)) {
    Dk = distance;
  }
  Dk = 1 - Dk / distance;
  return Dk;
}

// calculate radial gradient convergence for a single subpixel

float _c_calculate_rgc(int xM, int yM, __global float* imIntGx, __global float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity) {

    float vx, vy, Gx, Gy, dx, dy, distance, distanceWeight, GdotR, Dk;

    float xc = (xM + 0.5) / magnification;
    float yc = (yM + 0.5) / magnification;

    float RGC = 0;
    float distanceWeightSum = 0;

    int _start = -(int)(Gx_Gy_MAGNIFICATION * fwhm);
    int _end = (int)(Gx_Gy_MAGNIFICATION * fwhm + 1);

    for (int j = _start; j < _end; j++) {
        vy = (int)(Gx_Gy_MAGNIFICATION * yc) + j;
        vy /= Gx_Gy_MAGNIFICATION;

        if (0 < vy && vy <= rowsM - 1) {
            for (int i = _start; i < _end; i++) {
                vx = (int)(Gx_Gy_MAGNIFICATION * xc) + i;
                vx /= Gx_Gy_MAGNIFICATION;

                if (0 < vx && vx <= colsM - 1) {
                    dx = vx - xc;
                    dy = vy - yc;
                    distance = sqrt(dx * dx + dy * dy);

                    if (distance != 0 && distance <= tSO) {
                        Gx = imIntGx[(int)(vy * magnification * Gx_Gy_MAGNIFICATION * colsM * Gx_Gy_MAGNIFICATION) + (int)(vx * magnification * Gx_Gy_MAGNIFICATION)];
                        Gy = imIntGy[(int)(vy * magnification * Gx_Gy_MAGNIFICATION * colsM * Gx_Gy_MAGNIFICATION) + (int)(vx * magnification * Gx_Gy_MAGNIFICATION)];

                        distanceWeight = _c_calculate_dw(distance, tSS);
                        distanceWeightSum += distanceWeight;
                        GdotR = Gx*dx + Gy*dy;

                        if (GdotR < 0) {
                            Dk = _c_calculate_dk(Gx, Gy, dx, dy, distance);
                            RGC += Dk * distanceWeight;
                        }
                    }
                }
            }
        }
    }

    RGC /= distanceWeightSum;

    if (RGC >= 0 && sensitivity > 1) {
        RGC = pow(RGC, sensitivity);
    } else if (RGC < 0) {
        RGC = 0;
    }

    return RGC;
}


 __kernel void calculate_rgc(__global float* imIntGx, __global float* imIntGy, __global float* imInt, __global float* image_out, int nCols, int nRows, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity, int doIntensityWeighting) {

    // Index of the current pixel
    int f = get_global_id(0);
    int row = get_global_id(1);
    int col = get_global_id(2);

    // Output image dimensons
    int nPixels_out = nRows * nCols;

    // gradient image dimensions
    int nPixels_grad = nRows*Gx_Gy_MAGNIFICATION * nCols*Gx_Gy_MAGNIFICATION;

    row = row + magnification*2;
    col = col + magnification*2;

    if (doIntensityWeighting == 1) {
        image_out[f * nPixels_out + row * nCols + col] =  _c_calculate_rgc(col, row, &imIntGx[f * nPixels_grad], &imIntGy[f * nPixels_grad], nCols, nRows, magnification, Gx_Gy_MAGNIFICATION, fwhm, tSO, tSS, sensitivity) * imInt[f * nPixels_out + row * nCols + col];
    }
    else {
        image_out[f * nPixels_out + row * nCols + col] =  _c_calculate_rgc(col, row, &imIntGx[f * nPixels_grad], &imIntGy[f * nPixels_grad], nCols, nRows, magnification, Gx_Gy_MAGNIFICATION, fwhm, tSO, tSS, sensitivity);
    }
       }
