int _c_mandelbrot(float row, float col);int _c_mandelbrot(float row, float col);

// c2cl-define: MAX_ITERATIONS from _c_mandelbrot_benchmark.c
#define MAX_ITERATIONS 1000

// c2cl-define: DIVERGENCE from _c_mandelbrot_benchmark.c
#define DIVERGENCE 10

// c2cl-function: _c_mandelbrot from _c_mandelbrot_benchmark.c
int _c_mandelbrot(float row, float col) {
  float zrow = 0;
  float zcol = 0;
  float zrow_new, zcol_new;
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

__kernel void mandelbrot(__global int *output, float r_start, float r_end,
                         float c_start, float c_end) {

  const int r = get_global_id(0);
  const int c = get_global_id(1);
  const int rows = get_global_size(0);
  const int cols = get_global_size(1);

  float r_step = (r_end - r_start) / rows;
  float c_step = (c_end - c_start) / cols;

  float row = r_start + r * r_step;
  float col = c_start + c * c_step;
  // const int idx = r + c * rows;
  const int idx = c + r * cols;

  output[idx] = _c_mandelbrot(row, col);
}
