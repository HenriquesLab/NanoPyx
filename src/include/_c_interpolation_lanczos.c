#include <_c_interpolation_lanczos.h>

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
float _c_interpolate(float* image, float r, float c, int rows, int cols) {
  // return 0 if r OR c positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor((float) (r - 0.5));
  const int c_int = (int)floor((float) (c - 0.5));
  double v_interpolated = 0;

  double weight = 0;
  double weight_sum = 0;

  int r_neighbor, c_neighbor;
  double row_factor, col_factor;

  for (int j = 0; j <= TAPS; j++) {
    c_neighbor = c_int - HALF_TAPS + j;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }
    col_factor = _c_lanczos_kernel(c - (c_neighbor + 0.5));

    for (int i = 0; i <= TAPS; i++) {
      r_neighbor = r_int - HALF_TAPS + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }
      row_factor = _c_lanczos_kernel(r - (r_neighbor + 0.5));

      // Add the contribution from this tap to the interpolation
      weight = row_factor * col_factor;
      v_interpolated += image[r_neighbor * cols + c_neighbor] * weight;
      weight_sum += weight;
    }
  }
  return v_interpolated / weight_sum;
}