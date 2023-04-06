
// as in REF:
// https://github.com/HenriquesLab/NanoJ-eSRRF/blob/785c71b3bd508c938f63bb780cba47b0f1a5b2a7/resources/liveSRRF.cl
// under calculateGradient_2point
void gradient_roberts_cross(float *image, float *imGr, float *imGc, int rows,
                            int cols) {
  int r0, r1, c0, c1;
  for (int i = 1; i < rows; i++) {
    r0 = i * cols;
    r1 = (i - 1) * cols;
    for (int j = 1; j < cols; j++) {
      c0 = j;
      c1 = j - 1;
      imGr[r0 + j] = image[r1 + c1] - image[r1 + c0];
      imGc[r0 + j] = image[r1 + c1] - image[r0 + c1];
    }
  }
}

void gradient_2point(float *image, float *imGx, float *imGy, int h, int w) {
  int x0, y0, x1, y1;
  for (int j = 1; j < h; j++) {
    y1 = j * w;
    y0 = (j - 1) * w;
    for (int i = 1; i < w; i++) {
      x1 = i;
      x0 = i - 1;
      imGx[y1 + i] = image[y1 + x1] - image[y1 + x0];
      imGy[y1 + i] = image[y1 + x1] - image[y0 + x1];
    }
  }
}
