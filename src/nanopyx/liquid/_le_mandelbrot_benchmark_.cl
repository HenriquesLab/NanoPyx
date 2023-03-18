
// c2cl-define: MAX_ITERATIONS
#define MAX_ITERATIONS 1000
// c2cl-define: DIVERGENCE
#define DIVERGENCE 10

// c2cl-function: _c_mandelbrot
int _c_mandelbrot(double row, double col) {
  double zrow = 0;
  double zcol = 0;
  double zrow_new, zcol_new;
  int iterations = 0;
  while (zrow * zrow + zcol * zcol <= 4 && iterations < MAX_ITERATIONS) {
    zrow_new = zrow * zrow - zcol * zcol + row;
    zcol_new = 2 * zrow * zcol + col;
    zrow = zrow_new;
    zcol = zcol_new;
    iterations++;
  }
  return iterations;
}

__kernel void mandelbrot(__global int *output, double r_start, double r_end,
                         double c_start, double c_end) {

  const int r = get_global_id(0);
  const int c = get_global_id(1);
  const int rows = get_global_size(0);
  const int cols = get_global_size(1);

  double r_step = (r_end - r_start) / rows;
  double c_step = (c_end - c_start) / cols;

  double row = r_start + r * r_step;
  double col = c_start + c * c_step;
  // const int idx = r + c * rows;
  const int idx = c + r * cols;

  output[idx] = _c_mandelbrot(row, col);
}
