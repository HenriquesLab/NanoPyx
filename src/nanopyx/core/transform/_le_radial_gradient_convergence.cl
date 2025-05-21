float _c_calculate_rgc(int xM, int yM, __global float* imIntGx, __global float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity, float offset, float xyoffset, float angle);
double _c_calculate_dw(double distance, double tSS);
double _c_calculate_dk(float Gx, float Gy, float dx, float dy, float distance);

float2 _rotation_matrix(float2 point, float angle);

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

float2 _rotate_vector(float Gx, float Gy, float angle) {
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    float rotated_Gx = Gx * cos_angle - Gy * sin_angle;
    float rotated_Gy = Gx * sin_angle + Gy * cos_angle;
    return (float2)(rotated_Gx, rotated_Gy);
}

float2 _rotation_matrix(float2 point, float angle){

    //xcos - ysin
    //xsin + ycos

    float x = point.x;
    float y = point.y;

    return (float2)(x*cos(angle) - y*sin(angle), x*sin(angle) + y*cos(angle));
}

// calculate radial gradient convergence for a single subpixel

float _c_calculate_rgc(int xM, int yM, __global float* imIntGx, __global float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity, float offset, float xyoffset, float angle) {

    float vx, vy, Gx, Gy, dx, dy, distance, distanceWeight, GdotR, Dk;
    float2 correctedv;
    float2 correctedd;
    float correct_vx, correct_vy;

    float xc = (float)xM / magnification + offset; // offset in non-magnified space
    float yc = (float)yM / magnification + offset;

    float RGC = 0;
    float distanceWeightSum = 0;

    int _start = -(int)(fwhm);
    int _end = (int)(fwhm + 1);

    for (int j = _start; j < _end; j++) {
        vy = yc + j;

        if (0 < vy && vy <= (rowsM/magnification) - 1) {
            for (int i = _start; i < _end; i++) {
                vx = xc + i;

                if (0 < vx && vx <= (colsM/magnification) - 1) {
                    dx = vx - xc;
                    dy = vy - yc;
                    distance = sqrt(dx * dx + dy * dy);

                    if (distance != 0 && distance <= tSO) {

                        correct_vx = vx+xyoffset;
                        correct_vy = vy+xyoffset;

                        if (correct_vx<fabs(xyoffset)){

                            correct_vx = 0;

                        };

                        if (correct_vy<fabs(xyoffset)){

                            correct_vy = 0;

                        };

                        Gx = imIntGx[(int)((correct_vy) * magnification * Gx_Gy_MAGNIFICATION * colsM * Gx_Gy_MAGNIFICATION) + (int)((correct_vx) * magnification * Gx_Gy_MAGNIFICATION)];
                        Gy = imIntGy[(int)((correct_vy) * magnification * Gx_Gy_MAGNIFICATION * colsM * Gx_Gy_MAGNIFICATION) + (int)((correct_vx) * magnification * Gx_Gy_MAGNIFICATION)];

                        // Rotate the gradient components
                        float2 rotatedG = _rotate_vector(Gx, Gy, angle);
                        Gx = rotatedG.x;
                        Gy = rotatedG.y;

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


 __kernel void calculate_rgc(__global float* imIntGx, __global float* imIntGy, __global float* imInt, __global float* image_out, int nCols, int nRows, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity, int doIntensityWeighting, float offset, float xyoffset, float angle) {

    // Index of the current pixel
    int f = get_global_id(0);
    int row = get_global_id(1);
    int col = get_global_id(2);

    // Output image dimensons
    int nPixels_out = nRows * nCols;

    // gradient image dimensions
    int nPixels_grad = (int) (nRows*Gx_Gy_MAGNIFICATION * nCols*Gx_Gy_MAGNIFICATION);

    row = row + (int)(fwhm*magnification);
    col = col + (int)(fwhm*magnification);

    if (doIntensityWeighting == 1) {
        image_out[f * nPixels_out + row * nCols + col] =  _c_calculate_rgc(col, row, &imIntGx[f * nPixels_grad], &imIntGy[f * nPixels_grad], nCols, nRows, magnification, Gx_Gy_MAGNIFICATION, fwhm, tSO, tSS, sensitivity, offset, xyoffset, angle) * imInt[f * nPixels_out + row * nCols + col];
    }
    else {
        image_out[f * nPixels_out + row * nCols + col] =  _c_calculate_rgc(col, row, &imIntGx[f * nPixels_grad], &imIntGy[f * nPixels_grad], nCols, nRows, magnification, Gx_Gy_MAGNIFICATION, fwhm, tSO, tSS, sensitivity, offset, xyoffset, angle);
    }
       }
