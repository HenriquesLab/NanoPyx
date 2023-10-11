#include <math.h>

// bicubic interpolation
float _c_interpolate(float* image, float r, float c, int rows, int cols) {
  // return 0 if x OR y positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor((float) (r - 0.5));
  const int c_int = (int)floor((float) (c - 0.5));

  double a[4];
  double b[4];
  double dr = r - (r_int + 0.5);
  double dc = c - (c_int + 0.5);
  double dr2 = dr * dr;
  double dr3 = dr2 * dr;
  double dc2 = dc * dc;
  double dc3 = dc2 * dc;

  // Calculate the coefficients for the cubic polynomial
  a[0] = -0.5 * dc3 + dc2 - 0.5 * dc;
  a[1] = 1.5 * dc3 - 2.5 * dc2 + 1;
  a[2] = -1.5 * dc3 + 2 * dc2 + 0.5 * dc;
  a[3] = 0.5 * dc3 - 0.5 * dc2;

  b[0] = -0.5 * dr3 + dr2 - 0.5 * dr;
  b[1] = 1.5 * dr3 - 2.5 * dr2 + 1;
  b[2] = -1.5 * dr3 + 2 * dr2 + 0.5 * dr;
  b[3] = 0.5 * dr3 - 0.5 * dr2;

  double v_interpolated = 0;
  double weight = 0;
  double weight_sum = 0;

  int r_neighbor, c_neighbor;
  double row_factor, col_factor;

  for (int j = 0; j <= 3; j++) {
    c_neighbor = c_int - 1 + j;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }
    col_factor = a[j];

    for (int i = 0; i <= 3; i++) {
      r_neighbor = r_int - 1 + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }
      row_factor = b[i];

      weight = row_factor * col_factor;
      v_interpolated += image[r_neighbor * cols + c_neighbor] * weight;
      weight_sum += weight;
    }
  }
  return v_interpolated / weight_sum;
}