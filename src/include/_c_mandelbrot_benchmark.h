#ifndef _C_MANDELBROT_BENCHMARK_H
#define _C_MANDELBROT_BENCHMARK_H

// calculate a mandelbrot set
int _c_mandelbrot(double r, double c, int max_iter, double divergence) {
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

#endif
