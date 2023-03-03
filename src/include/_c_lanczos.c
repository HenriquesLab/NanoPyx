#include "_c_lanczos.h"

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
