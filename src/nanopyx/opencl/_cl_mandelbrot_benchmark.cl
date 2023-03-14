
// copy of _c_mandelbrot() from _c_mandelbrot_benchmark.h
int _c_mandelbrot(float r, float c, int max_iter, float divergence) {
  double zr = 0.0;
  double zi = 0.0;
  double zr2 = 0.0;
  double zi2 = 0.0;
  int i = 0;
  while (i < max_iter && zr2 + zi2 < divergence) {
    zi = 2.0 * zr * zi + c / 500 - 1;
    zr = zr2 - zi2 + r / 500 - 1.5;
    zr2 = zr * zr;
    zi2 = zi * zi;
    i++;
  }
  return i;
}

__kernel void mandelbrot(__global int *output, const int max_iter,
                         const double divergence) {

  const int r = get_global_id(0);
  const int c = get_global_id(1);
  const int rows = get_global_size(0);
  const int idx = r + c * rows;

  output[idx] = _c_mandelbrot(r, c, max_iter, divergence);
}
