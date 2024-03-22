#include <math.h>

void _c_gradient_radiality(float* image, float* imGc, float* imGr, int rows,
                          int cols) {
  int c0, c_m1, c_p1, r0, r_m1, r_p1;

  // for j in range(1, h-1):
  //   for i in range(1, w-1):
  //       imGx[j,i] = -imRaw[j,i-1]+imRaw[j,i+1]
  //       imGy[j,i] = -imRaw[j-1,i]+imRaw[j+1,i]

  for (int j = 1; j < rows - 1; j++) {
    r0 = j * cols;
    r_m1 = (j - 1) * cols;
    r_p1 = (j + 1) * cols;

    for (int i = 1; i < cols - 1; i++) {
      c0 = i;
      c_m1 = i - 1;
      c_p1 = i + 1;
      imGc[r0 + c0] = -image[r0 + c_m1] + image[r0 + c_p1];
      imGr[r0 + c0] = -image[r_m1 + c0] + image[r_p1 + c0];
    }
  }
}

// as in REF:
// https://github.com/HenriquesLab/NanoJ-eSRRF/blob/785c71b3bd508c938f63bb780cba47b0f1a5b2a7/resources/liveSRRF.cl
// under calculateGradient_2point
void _c_gradient_2point(float* image, float* imGc, float* imGr, int rows,
                     int cols) {
  int c0, r0, c1, r1;
  for (int j = 1; j < rows; j++) {
    r1 = j * cols;
    r0 = (j - 1) * cols;
    for (int i = 1; i < cols; i++) {
      c1 = i;
      c0 = i - 1;
      imGc[r1 + i] = image[r1 + c1] - image[r1 + c0];
      imGr[r1 + i] = image[r1 + c1] - image[r0 + c1];
    }
  }
}

// https://github.com/HenriquesLab/NanoJ-eSRRF/blob/785c71b3bd508c938f63bb780cba47b0f1a5b2a7/resources/liveSRRF.cl
// under calculateGradientRobX
void _c_gradient_roberts_cross(float* image, float* imGc, float* imGr, int rows, int cols) {
    int c1, r1, c0, r0;
    float im_c0_r1, im_c1_r0, im_c0_r0, im_c1_r1;

    for (r1 = 0; r1 < rows; r1++) {
        for (c1 = 0; c1 < cols; c1++) {

            c0 = c1 > 0 ? c1 - 1 : 0;
            r0 = r1 > 0 ? r1 - 1 : 0;

            im_c0_r1 = image[r0 * cols + c1];
            im_c1_r0 = image[r1 * cols + c0];
            im_c0_r0 = image[r0 * cols + c0];
            im_c1_r1 = image[r1 * cols + c1];

            imGc[r1 * cols + c1] = im_c0_r1 - im_c1_r0 + im_c1_r1 - im_c0_r0;
            imGr[r1 * cols + c1] = -im_c0_r1 + im_c1_r0 + im_c1_r1 - im_c0_r0;
        }
    }
}

// as in https://www.nature.com/articles/s41592-022-01669-y#MOESM1
// 3D Gradient calculation
void _c_gradient_3d(float* image, float* imGc, float* imGr, float* imGs, int slices,
                 int rows, int cols) {
  float ip0, ip1, ip2, ip3, ip4, ip5, ip6, ip7;

  int z_i, y_i, x_i, z_1, y_1, x_1;

  for (z_i = 0; z_i < slices; z_i++) {
    for (y_i = 0; y_i < rows; y_i++) {
      for (x_i = 0; x_i < cols; x_i++) {

        z_1 = z_i >= slices - 1 ? slices - 1 : z_i + 1;
        y_1 = y_i >= rows - 1 ? rows - 1 : y_i + 1;
        x_1 = x_i >= cols - 1 ? cols - 1 : x_i + 1;
        ip0 = image[z_i * rows * cols + y_i * cols + x_i];
        ip1 = image[z_i * rows * cols + y_i * cols + x_1];
        ip2 = image[z_i * rows * cols + y_1 * cols + x_i];
        ip3 = image[z_i * rows * cols + y_1 * cols + x_1];
        ip4 = image[z_1 * rows * cols + y_i * cols + x_i];
        ip5 = image[z_1 * rows * cols + y_i * cols + x_1];
        ip6 = image[z_1 * rows * cols + y_1 * cols + x_i];
        ip7 = image[z_1 * rows * cols + y_1 * cols + x_1];
        imGc[z_i * rows* cols + y_i * cols + x_i] =
            (ip1 + ip3 + ip5 + ip7 - ip0 - ip2 - ip4 - ip6) / 4;
        imGr[z_i * rows* cols + y_i * cols + x_i] =
            (ip2 + ip3 + ip6 + ip7 - ip0 - ip1 - ip4 - ip5) / 4;
        imGs[z_i * rows* cols + y_i * cols + x_i] =
            (ip4 + ip5 + ip6 + ip7 - ip0 - ip1 - ip2 - ip3) / 4;
      }
    }
  }
}

void _c_gradient_2_point_3d(float* image, float* imGc, float* imGr, float* imGs, int slices, int rows, int cols) {
  int s_i, y_i, x_i, s0, y0, x0;

  for (s_i=0; s_i<slices; s_i++) {
    for (y_i=0; y_i<rows; y_i++) {
      for (x_i=0; x_i<cols; x_i++) {
        s0 = s_i > 0 ? s_i - 1 : 0;
        y0 = y_i > 0 ? y_i - 1 : 0;
        x0 = x_i > 0 ? x_i - 1 : 0;
        imGc[s_i*rows*cols + y_i*cols + x_i] = image[s_i*rows*cols + y_i*cols + x_i] - image[s_i*rows*cols + y_i*cols + x0];
        imGr[s_i*rows*cols + y_i*cols + x_i] = image[s_i*rows*cols + y_i*cols + x_i] - image[s_i*rows*cols + y0*cols + x_i];
        imGs[s_i*rows*cols + y_i*cols + x_i] = image[s_i*rows*cols + y_i*cols + x_i] - image[s0*rows*cols + y_i*cols + x_i];

      }
    }
  }
}