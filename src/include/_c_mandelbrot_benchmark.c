
#define MAX_ITERATIONS 1000
#define DIVERGENCE 10

int _c_mandelbrot(double row, double col) {
  double zrow = 0;
  double zcol = 0;
  double zrow_new, zcol_new;
  int iterations = 0;
  while (zrow * zrow + zcol * zcol <= DIVERGENCE &&
         iterations < MAX_ITERATIONS) {
    zrow_new = zrow * zrow - zcol * zcol + row;
    zcol_new = 2 * zrow * zcol + col;
    zrow = zrow_new;
    zcol = zcol_new;
    iterations++;
  }
  return iterations;
}
