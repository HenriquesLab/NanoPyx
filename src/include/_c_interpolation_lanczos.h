#ifndef _C_INTERPOLATION_LANCZOS_H
#define _C_INTERPOLATION_LANCZOS_H

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Calculate the Lanczos kernel (windowed sinc function) value for a given
// value. REF: https://en.wikipedia.org/wiki/Lanczos_resampling
double _c_lanczos_kernel(double v, int taps) {
  if (v == 0) {
    return 1.0;
  } else if (fabs(v) < taps) {
    return taps * sin(M_PI * v) * sin(M_PI * v / taps) / (M_PI * M_PI * v * v);
  } else {
    return 0.0;
  }
}

#endif  // _C_INTERPOLATION_LANCZOS_H
