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

void _rotate_vector(float* Gx, float* Gy, float angle) {
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);

    float original_Gx = *Gx;
    float original_Gy = *Gy;

    *Gx = original_Gx * cos_angle - original_Gy * sin_angle;
    *Gy = original_Gx * sin_angle + original_Gy * cos_angle;
}

float _c_calculate_rgc(int xM, int yM, float* imIntGx, float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity, float offset, float xyoffset, float angle) {

    float vx, vy, Gx, Gy, dx, dy, distance, distanceWeight, GdotR, Dk;
    float correct_vx, correct_vy;

    float xc = (float)xM / magnification + offset; // offset in non-magnified space
    float yc = (float)yM / magnification + offset;

    float RGC = 0;
    float distanceWeightSum = 0;

    int _start = -(int)(fwhm);
    int _end = (int)(fwhm + 1);

    for (int j = _start; j < _end; j++) {
        vy = yc + j;

        if (0 < vy && vy <= rowsM/magnification - 1) {
            for (int i = _start; i < _end; i++) {
                vx = xc + i;

                if (0 < vx && vx <= colsM/magnification - 1) {
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

                        _rotate_vector(&Gx, &Gy, angle);

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
  float Dk = sqrt((Gy * dz - Gz * dy) * (Gy * dz - Gz * dy) + (Gz * dx - Gx * dz) * (Gz * dx - Gx * dz) + (Gx * dy - Gy * dx) * (Gx * dy - Gy * dx)) / sqrt(Gx * Gx + Gy * Gy + Gz * Gz);
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

float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float voxel_ratio, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) {

    float vx, vy, vz, Gx, Gy, Gz, dx, dy, dz, distance, distance_xy, distance_z, distanceWeight, distanceWeight_xy, distanceWeight_z, GdotR, Dk;

    float xc = (float)(xM) / magnification_xy;
    float yc = (float)(yM) / magnification_xy;
    float zc = (float)(sliceM) / magnification_z;

    float RGC = 0;
    float distanceWeightSum = 0;
    float distanceWeightSum_xy = 0;
    float distanceWeightSum_z = 0;

    int _start = -(int)(fwhm);
    int _end = (int)(fwhm + 1);

    int _start_z = -(int)(fwhm_z);
    int _end_z = (int)(fwhm_z + 1);
    
    for (int j = _start; j <= _end; j++) {
        vy = yc + j;

        if (0 < vy && vy <= (rowsM/magnification_xy) - 1) {
            for (int i = _start; i <= _end; i++) {
                vx = xc + i;

                if (0 < vx && vx <= (colsM/magnification_xy) - 1) {
                    for (int k = _start_z; k <= _end_z; k++) {
                        vz = zc + k;

                        if (0 < vz && vz <= (slicesM/magnification_z) - 1) {
                            dx = vx - xc;
                            dy = vy - yc;
                            distance_z = vz - zc;
                            distance = sqrt(dx * dx + dy * dy + distance_z * distance_z);
                            distance_xy = sqrt(dx * dx + dy * dy);
                            
                            if (distance != 0 && distance_xy <= tSO && distance_z <= tSO_z) {
                                int linear_index = (int)(vz * magnification_z) * rowsM * colsM +
                                                   (int)(magnification_xy * vy) * colsM +
                                                   (int)(magnification_xy * vx);

                                Gx = imIntGx[linear_index];
                                Gy = imIntGy[linear_index];
                                Gz = imIntGz[linear_index];

                                distanceWeight = _c_calculate_dw3D(distance, distance_xy, distance_z, tSS, tSS_z);
                                
                                distanceWeightSum += distanceWeight;
                                GdotR = Gx*dx + Gy*dy + Gz*distance_z;

                                if (GdotR < 0) {
                                    Dk = _c_calculate_dk3D(Gx, Gy, Gz, dx, dy, distance_z, distance);
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

    if (RGC >= 0 && sensitivity > 1) {
        RGC = pow(RGC, sensitivity);
    } else if (RGC < 0) {
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
