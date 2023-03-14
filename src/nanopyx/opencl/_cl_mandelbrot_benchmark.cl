__kernel void mandelbrot(__global int *output, const int max_iter,
                         const double divergence) {

  const int r = get_global_id(0);
  const int c = get_global_id(1);
  const int rows = get_global_size(0);

  double real = 1.5 * (r - 500) / (0.5 * 1000);
  double imag = (c - 500) / (0.5 * 1000);

  double real2 = real;
  double imag2 = imag;

  for (int i = 0; i < max_iter; i++) {

    real2 = real2 * real2 - imag2 * imag2 + real;
    imag2 = 2 * real2 * imag2 + imag;

    if (real2 * real2 + imag2 * imag2 > divergence) {
      output[r + c * rows] = i;
      return;
    }
  }
}
