#define TAPS 4

#include <math.h>

// Lanczos function used in Lanczos interpolation
double _c_lanczos_kernel(double v) {
  if (v == 0) {
    return 1.0;
  } else if (fabs(v) < TAPS) {
    double v_pi = v * M_PI;
    return TAPS * sin(v_pi) * sin(v_pi / TAPS) / (v_pi * v_pi);
  } else {
    return 0.0;
  }
}

// Lanczos interpolation
float _c_interpolate(float *image, float r, float c, int rows, int cols) {
  // return 0 if x OR y positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  // Determine the low and high indices for the row and column dimensions
  const int r_low = (int)fmax(floor(r) - TAPS, 0);
  const int r_high = (int)fmin(ceil(r) + TAPS + 1, rows);
  const int c_low = (int)fmax(floor(c) - TAPS + 1, 0);
  const int c_high = (int)fmin(ceil(c) + TAPS, cols);

  double interpolated_value = 0;
  double weight = 0;
  double weight_sum = 0;

  double row_factor, col_factor;

  for (int r_neighbor = r_low; r_neighbor <= r_high; r_neighbor++) {
    row_factor = _c_lanczos_kernel(r - r_neighbor);
    for (int c_neighbor = c_low; c_neighbor <= c_high; c_neighbor++) {
      col_factor = _c_lanczos_kernel(c - c_neighbor);

      // Add the contribution from this tap to the interpolation
      weight = row_factor * col_factor;
      interpolated_value += image[r_neighbor * cols + c_neighbor] * weight;
      weight_sum += weight;
    }
  }
  return interpolated_value / weight_sum;
}
