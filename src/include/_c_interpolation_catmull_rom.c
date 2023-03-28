
#include <math.h>

// Cubic function used in Catmull-Rom interpolation
// https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull.E2.80.93Rom_spline
float _c_cubic(float v) {
  float a = 0.5;
  float z = 0;
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
  // return 0 if x OR y positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int u0 = (int)floor(r - 0.5);
  const int v0 = (int)floor(c - 0.5);
  float q = 0;
  float p = 0;

  int u, v;

  for (int j = 0; j < 4; j++) {
    v = v0 - 1 + j;
    p = 0;
    if (v < 0 || v >= cols) {
      continue;
    }

    for (int i = 0; i < 4; i++) {
      u = u0 - 1 + i;
      if (u < 0 || u >= rows) {
        continue;
      }
      p = p + image[u * cols + v] * _c_cubic(r - (u + 0.5));
    }
    q = q + p * _c_cubic(c - (v + 0.5));
  }
  return q;
}
