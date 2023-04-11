
// as in REF:
// https://github.com/HenriquesLab/NanoJ-eSRRF/blob/785c71b3bd508c938f63bb780cba47b0f1a5b2a7/resources/liveSRRF.cl
// under calculateGradient_2point
void gradient_roberts_cross(float *image, float *imGr, float *imGc, int rows,
                            int cols) { //TODO: write the Roberts cross gradient (this is 2point)
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

// as in https://www.nature.com/articles/s41592-022-01669-y#MOESM1
// 3D Gradient calculation

void gradient_3d(float* image, float* imGx, float* imGy, float* imGz, int d, int h, int w) {
    float ip0, ip1, ip2, ip3, ip4, ip5, ip6, ip7;

    int z_i, y_i, x_i;

    for (z_i = 0; z_i < d-1; z_i++) {
        for (y_i = 0; y_i < h-1; y_i++) {
            for (x_i = 0; x_i < w-1; x_i++) {
                ip0 = image[z_i*h*w + y_i*w + x_i];
                ip1 = image[z_i*h*w + y_i*w + x_i + 1];
                ip2 = image[z_i*h*w + (y_i+1)*w + x_i];
                ip3 = image[z_i*h*w + (y_i+1)*w + x_i + 1];
                ip4 = image[(z_i+1)*h*w + y_i*w + x_i];
                ip5 = image[(z_i+1)*h*w + y_i*w + x_i + 1];
                ip6 = image[(z_i+1)*h*w + (y_i+1)*w + x_i];
                ip7 = image[(z_i+1)*h*w + (y_i+1)*w + x_i + 1];
                imGx[z_i*h*w + y_i*w + x_i] = (ip1 + ip3 + ip5 + ip7 - ip0 - ip2 - ip4 - ip6) / 4;
                imGy[z_i*h*w + y_i*w + x_i] = (ip2 + ip3 + ip6 + ip7 - ip0 - ip1 - ip4 - ip5) / 4;
                imGz[z_i*h*w + y_i*w + x_i] = (ip4 + ip5 + ip6 + ip7 - ip0 - ip1 - ip2 - ip3) / 4;
            }
        }
    }
}
