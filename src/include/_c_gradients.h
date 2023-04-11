#ifndef _C_GRADIENTS_H
#define _C_GRADIENTS_H

void gradient_roberts_cross(float *image, float *imGr, float *imGc, int rows,
                            int cols);

void gradient_2point(float *image, float *imGx, float *imGy, int h, int w);

void gradient_3d(float* image, float* imGx, float* imGy, float* imGz, int d, int h, int w);

#endif
