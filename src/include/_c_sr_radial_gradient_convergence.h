#ifndef _C_SR_RADIAL_GRADIENT_CONVERGENCE_H
#define _C_SR_RADIAL_GRADIENT_CONVERGENCE_H

#include <math.h>

double _c_calculate_dw(double distance, double tSS);

double _c_calculate_dk(float Gx, float Gy, float dx, float dy, float distance);

float _c_calculate_rgc(int xM, int yM, float* imIntGx, float* imIntGy, float* imInt, int w, int h, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity);

#endif
