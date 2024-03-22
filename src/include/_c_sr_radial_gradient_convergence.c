#include <math.h>
#include <stdio.h>

double _c_calculate_dw(double distance, double tSS) {
  return pow((distance * exp((-distance * distance) / tSS)), 4);
}

double _c_calculate_dk(float Gx, float Gy, float dx, float dy, float distance) {
  float Dk = fabs(Gy * dx - Gx * dy) / sqrt(Gx * Gx + Gy * Gy);
  if (isnan(Dk)) {
    Dk = distance;
  }
  Dk = 1 - Dk / distance;
  return Dk;
}

float _c_calculate_rgc(int xM, int yM, float* imIntGx, float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity) {

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

double _c_calculate_dw3D_isotropic(double distance, double tSS) {
  return pow((distance * exp(-(distance * distance) / tSS)), 4);
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
  float Dk = sqrtf((Gy * dz - Gz * dy) * (Gy * dz - Gz * dy) + (Gz * dx - Gx * dz) * (Gz * dx - Gx * dz) + (Gx * dy - Gy * dx) * (Gx * dy - Gy * dx)) / sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
  if (isnan(Dk)) {
    Dk = distance;
  }
  Dk = 1 - Dk / distance;
  return Dk;
}

float _c_get_bound_value(float* im, int slices, int rows, int cols, int s, int r, int c){
    int _s = s > 0 ? s : 0;
    _s = _s < slices - 1 ? _s : slices - 1;
    int _r = r > 0 ? r : 0;
    _r = _r < rows - 1 ? _r : rows - 1;
    int _c = c > 0 ? c : 0;
    _c = _c < cols - 1 ? _c : cols - 1;

    return im[_s * rows * cols + _r * cols + _c];
}

float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float ratio_px, float Gx_Gy_MAGNIFICATION, float Gz_MAGNIFICATION, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) {

    float vx, vy, vz, Gx, Gy, Gz, dx, dy, dz, dz_real, distance, distance_xy, distance_z, distanceWeight, distanceWeight_xy, distanceWeight_z, GdotR, Dk;

    float xc = (xM + 0.5) / magnification_xy;
    float yc = (yM + 0.5) / magnification_xy;
    float zc = (sliceM + 0.5) / magnification_z;

    float RGC = 0;
    float distanceWeightSum = 0;
    float distanceWeightSum_xy = 0;
    float distanceWeightSum_z = 0;

    int _start = -(int)(Gx_Gy_MAGNIFICATION * fwhm);
    int _end = (int)(Gx_Gy_MAGNIFICATION * fwhm + 1);

    int _start_z = -(int)(Gz_MAGNIFICATION * fwhm_z);
    int _end_z = (int)(Gz_MAGNIFICATION * fwhm_z + 1);
    
    for (int j = _start; j <= _end; j++) {
        vy = ((float) ((int) (Gx_Gy_MAGNIFICATION*yc)) + j)/(float) Gx_Gy_MAGNIFICATION;

        if (0 < vy && vy < rowsM - 1) {
            for (int i = _start; i <= _end; i++) {
                vx = ((float) ((int) (Gx_Gy_MAGNIFICATION*xc)) + i)/(float) Gx_Gy_MAGNIFICATION;

                if (0 < vx && vx < colsM - 1) {
                    for (int k = _start_z; k <= _end_z; k++) {
                        vz = ((float) ((int) (Gz_MAGNIFICATION*zc)) + k)/(float) Gz_MAGNIFICATION;

                        if (0 < vz && vz < slicesM - 1) {
                            dx = vx - xc;
                            dy = vy - yc;
                            dz = vz - zc; 
                            dz_real = dz * ratio_px; // This has been already divided by magnification_z
                            distance = sqrt(dx * dx + dy * dy + dz_real * dz_real);
                            distance_xy = sqrt(dx * dx + dy * dy);
                            distance_z = dz_real;
                            
                            if (distance != 0 && distance_xy <= tSO && distance_z <= tSO_z) {
                                Gx = _c_get_bound_value(imIntGx, slicesM*Gz_MAGNIFICATION, rowsM*Gx_Gy_MAGNIFICATION, colsM*Gx_Gy_MAGNIFICATION, vz * magnification_z * Gz_MAGNIFICATION, magnification_xy * Gx_Gy_MAGNIFICATION * vy, magnification_xy * Gx_Gy_MAGNIFICATION * vx);
                                Gy = _c_get_bound_value(imIntGy, slicesM*Gz_MAGNIFICATION, rowsM*Gx_Gy_MAGNIFICATION, colsM*Gx_Gy_MAGNIFICATION, vz * magnification_z * Gz_MAGNIFICATION, magnification_xy * Gx_Gy_MAGNIFICATION * vy, magnification_xy * Gx_Gy_MAGNIFICATION * vx);
                                Gz = _c_get_bound_value(imIntGz, slicesM*Gz_MAGNIFICATION, rowsM*Gx_Gy_MAGNIFICATION, colsM*Gx_Gy_MAGNIFICATION, vz * magnification_z * Gz_MAGNIFICATION, magnification_xy * Gx_Gy_MAGNIFICATION * vy, magnification_xy * Gx_Gy_MAGNIFICATION * vx);

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

float _c_calculate_dk_3d(float dx, float dy, float dz, float Gx, float Gy, float Gz, float Gmag, float distance) {
    float G1 = dy*Gz - dz*Gy;
    float G2 = dz*Gx - dx*Gz;
    float G3 = dx*Gy - dy*Gx;
    float cross_product = sqrt(G1*G1 + G2*G2 + G3*G3)/Gmag;

    if (isnan(cross_product)) {
        cross_product = distance;
    }

    cross_product = 1 - cross_product / distance;

    return cross_product;
}
