void _c_gradient_3d(float* image, float* imGc, float* imGr, float* imGs, int slices,
                 int rows, int cols) {
  float ip0, ip1, ip2, ip3, ip4, ip5, ip6, ip7;

  int z_i, y_i, x_i, z_1, y_1, x_1;

  for (z_i = 0; z_i < slices; z_i++) {
    for (y_i = 0; y_i < rows; y_i++) {
      for (x_i = 0; x_i < cols; x_i++) {

        z_1 = z_i >= slices - 1 ? slices - 1 : z_i + 1;
        y_1 = y_i >= rows - 1 ? rows - 1 : y_i + 1;
        x_1 = x_i >= cols - 1 ? cols - 1 : x_i + 1;
        ip0 = image[z_i * rows * cols + y_i * cols + x_i];
        ip1 = image[z_i * rows * cols + y_i * cols + x_1];
        ip2 = image[z_i * rows * cols + y_1 * cols + x_i];
        ip3 = image[z_i * rows * cols + y_1 * cols + x_1];
        ip4 = image[z_1 * rows * cols + y_i * cols + x_i];
        ip5 = image[z_1 * rows * cols + y_i * cols + x_1];
        ip6 = image[z_1 * rows * cols + y_1 * cols + x_i];
        ip7 = image[z_1 * rows * cols + y_1 * cols + x_1];
        imGc[z_i * rows* cols + y_i * cols + x_i] =
            (ip1 + ip3 + ip5 + ip7 - ip0 - ip2 - ip4 - ip6) / 4;
        imGr[z_i * rows* cols + y_i * cols + x_i] =
            (ip2 + ip3 + ip6 + ip7 - ip0 - ip1 - ip4 - ip5) / 4;
        imGs[z_i * rows* cols + y_i * cols + x_i] =
            (ip4 + ip5 + ip6 + ip7 - ip0 - ip1 - ip2 - ip3) / 4;
      }
    }
  }
}

double _c_calculate_dw3D(double distance, double distance_xy, double distance_z,double tSS, double tSS_z) {
  float D_weight_xy, D_weight_z, D_weight;
  D_weight_xy = (exp((-distance_xy * distance_xy) / tSS));
  D_weight_z = (exp((-distance_z * distance_z) / tSS_z));
  D_weight = distance * (D_weight_xy * D_weight_z);
  return pow(D_weight, 4);
}

double _c_calculate_dw_xy(double distance_xy, double tSS) {
  return pow((distance_xy * exp((-distance_xy * distance_xy) / tSS)), 4);
}

double _c_calculate_dw_z(double distance_z, double tSS_z) {
  return pow((distance_z * exp((-distance_z * distance_z) / tSS_z)), 4);
}

double _c_calculate_dk3D(float Gx, float Gy, float Gz, float dx, float dy, float dz, float distance) {
  float Dk = sqrt((Gy * dz - Gz * dy) * (Gy * dz - Gz * dy) + (Gz * dx - Gx * dz) * (Gz * dx - Gx * dz) + (Gx * dy - Gy * dx) * (Gx * dy - Gy * dx)) / sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
  if (isnan(Dk)) {
    Dk = distance;
  }
  Dk = 1 - Dk / distance;
  return Dk;
}

float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float ratio_px, float Gx_Gy_MAGNIFICATION, float Gz_MAGNIFICATION, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) {

    float vx, vy, vz, Gx, Gy, Gz, dx, dy, dz, dz_real, distance, distance_xy, distance_z, distanceWeight, distanceWeight_xy, distanceWeight_z, GdotR, Dk;

    float xc = (xM) / magnification_xy;
    float yc = (yM) / magnification_xy;
    float zc = (sliceM) / magnification_z;

    float RGC = 0;
    float distanceWeightSum = 0;
    float distanceWeightSum_xy = 0;
    float distanceWeightSum_z = 0;

    int _start = -(int)(2 * fwhm);
    int _end = (int)(2 * fwhm + 1);

    int _start_z = -(int)(2 * fwhm_z);
    int _end_z = (int)(2 * fwhm_z + 1);
    
    for (int j = _start; j <= _end; j++) {
        vy = yc + j;

        if (0 < vy && vy < rowsM/magnification_xy - 1) {
            for (int i = _start; i <= _end; i++) {
                vx = xc + i;

                if (0 < vx && vx < colsM/magnification_xy - 1) {
                    for (int k = _start_z; k <= _end_z; k++) {
                        vz = zc + k;

                        if (0 < vz && vz < slicesM/magnification_z - 1) {
                            dx = vx - xc;
                            dy = vy - yc;
                            dz = vz - zc; 
                            dz_real = dz * ratio_px; // This has been already divided by magnification_z
                            distance = sqrt(dx * dx + dy * dy + dz_real * dz_real);
                            distance_xy = sqrt(dx * dx + dy * dy);
                            distance_z = dz_real;
                            
                            if (distance != 0 && distance_xy <= tSO && distance_z <= tSO_z) {
                                int linear_index = (int)(vz * magnification_z * Gz_MAGNIFICATION) * rowsM * Gx_Gy_MAGNIFICATION * colsM * Gx_Gy_MAGNIFICATION +
                                                   (int)(magnification_xy * Gx_Gy_MAGNIFICATION * vy) * colsM * Gx_Gy_MAGNIFICATION +
                                                   (int)(magnification_xy * Gx_Gy_MAGNIFICATION * vx);

                                Gx = imIntGx[linear_index];
                                Gy = imIntGy[linear_index];
                                Gz = imIntGz[linear_index];

                                // distanceWeight = _c_calculate_dw3D_isotropic(distance, tSS);
                                distanceWeight = _c_calculate_dw3D(distance, distance_xy, distance_z, tSS, tSS_z);

                                // distanceWeight_xy = _c_calculate_dw_xy(distance_xy, tSS); 
                                // distanceWeight_z = _c_calculate_dw_z(distance_z, tSS_z);
                                
                                distanceWeightSum += distanceWeight;
                                // distanceWeightSum_xy += distanceWeight_xy;
                                // distanceWeightSum_z += distanceWeight_z;
                                GdotR = Gx*dx + Gy*dy + Gz*dz_real;

                                if (GdotR < 0) {
                                    Dk = _c_calculate_dk3D(Gx, Gy, Gz, dx, dy, dz_real, distance);
                                    RGC += (Dk * distanceWeight);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    RGC /= distanceWeightSum;

    if (RGC >= 0) {
        RGC = pow(RGC, sensitivity);
    } else {
        RGC = 0;
    }

    return RGC;
}

