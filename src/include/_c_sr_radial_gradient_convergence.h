#ifndef _C_SR_RADIAL_GRADIENT_CONVERGENCE_H
#define _C_SR_RADIAL_GRADIENT_CONVERGENCE_H

#include <math.h>

double _c_calculate_dw(double distance, double tSS);

double _c_calculate_dk(float Gx, float Gy, float dx, float dy, float distance);

#endif
