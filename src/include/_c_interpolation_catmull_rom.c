
#include <math.h>

// Cubic function used in Catmull-Rom interpolation
// https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull.E2.80.93Rom_spline
double _c_cubic(double v) {
  double a = 0.5;
  double z = 0;
  if (v < 0) {
    v = -v;
  }
  if (v < 1) {
    z = v * v * (v * (-a + 2) + (a - 3)) + 1;
  } else if (v < 2) {
    z = -a * v * v * v + 5 * a * v * v - 8 * a * v + 4 * a;
  }
  return z;
}

// Catmull-Rom interpolation
double _c_interpolate(float* image, double r, double c, int rows, int cols) {
  // return 0 if x OR y positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r0 = (int)r;
  const int c0 = (int)c;

  const int u0 = (int)floor(r - 0.5);
  const int v0 = (int)floor(c - 0.5);
  double q = 0;
  double p = 0;

  int u, v;

  for (int j = 0; j < 4; j++) {
    v = v0 - 1 + j;
    p = 0;
    for (int i = 0; i < 4; i++) {
      u = u0 - 1 + i;
      int _u = (int)fmax(0, fmin(u, rows - 1));
      int _v = (int)fmax(0, fmin(v, cols - 1));
      p = p + image[_u * cols + _v] * _c_cubic(r - (u + 0.5));
    }
    q = q + p * _c_cubic(c - (v + 0.5));
  }
  return q;
}
