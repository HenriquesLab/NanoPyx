
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
float _c_interpolate(float* image, float r, float c, int rows, int cols) {
  // return 0 if r OR c positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor((float) (r - 0.5));
  const int c_int = (int)floor((float) (c - 0.5));
  double q = 0;
  double p = 0;

  int r_neighbor, c_neighbor;

  for (int j = 0; j < 4; j++) {
    c_neighbor = c_int - 1 + j;
    p = 0;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }

    for (int i = 0; i < 4; i++) {
      r_neighbor = r_int - 1 + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }
      p = p + image[r_neighbor * cols + c_neighbor] *
                  _c_cubic(r - (r_neighbor + 0.5));
    }
    q = q + p * _c_cubic(c - (c_neighbor + 0.5));
  }
  return q;
}
