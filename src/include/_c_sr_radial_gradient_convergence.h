#ifndef _C_SR_RADIAL_GRADIENT_CONVERGENCE_H
#define _C_SR_RADIAL_GRADIENT_CONVERGENCE_H

#include <math.h>

double _c_calculate_dw(double distance, double tSS);

double _c_calculate_dk(float Gx, float Gy, float dx, float dy, float distance);

float _c_calculate_rgc(int xM, int yM, float* imIntGx, float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity);

double _c_calculate_dk3D(float Gx, float Gy, float Gz, float dx, float dy, float dz, float distance);

float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification, float Gx_Gy_MAGNIFICATION, float Gz_MAGNIFICATION, float fwhm, float fwhm_z, float tSO, float tSS, float sensitivity);


#endif
