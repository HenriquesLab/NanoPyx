#ifndef _C_INTERPOLATION_LANCZOS_H
#define _C_INTERPOLATION_LANCZOS_H

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265359f
// #define PI 3.14159265358979323846
#endif

#define TAPS 4
#define HALF_TAPS 2

double _c_lanczos_kernel(double v);
float _c_interpolate(float *image, float r, float c, int rows, int cols);

#endif  // _C_INTERPOLATION_LANCZOS_H
