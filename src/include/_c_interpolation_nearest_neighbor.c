
float _c_interpolate(float *image, float row, float col, int rows, int cols) {
  int r = (int)row;
  int c = (int)col;
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }
  return image[r * cols + c];
}
