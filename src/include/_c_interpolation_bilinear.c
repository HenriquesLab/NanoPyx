#include <math.h>

// bilinear interpolation
float _c_interpolate(float* image, float r, float c, int rows, int cols) {
  // return 0 if r OR c positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor(r - 0.5);
  const int c_int = (int)floor(c - 0.5);

  double dr = r - (r_int + 0.5);
  double dc = c - (c_int + 0.5);

  double v_interpolated = 0;

  int r_neighbor, c_neighbor;

  for (int j = 0; j <= 1; j++) {
    c_neighbor = c_int + j;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }

    for (int i = 0; i <= 1; i++) {
      r_neighbor = r_int + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }

      v_interpolated += image[r_neighbor * cols + c_neighbor] *
                        (1 - fabs(dr - i)) * (1 - fabs(dc - j));
    }
  }
  return v_interpolated;
}