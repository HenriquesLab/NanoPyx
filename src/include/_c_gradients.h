#ifndef _C_GRADIENTS_H
#define _C_GRADIENTS_H

void _c_gradient_radiality(float* image, float* imGc, float* imGr, int rows,
                           int cols);

void _c_gradient_2point(float* image, float* imGc, float* imGr, int rows,
                     int cols);

void _c_gradient_roberts_cross(float* image, float* imGc, float* imGr, int rows, int cols);

void _c_gradient_3d(float* image, float* imGc, float* imGr, float* imGs, int slices,
                 int rows, int cols);

void _c_gradient_2_point_3d(float* image, float* imGc, float* imGr, float* imGs, int slices, int rows, int cols);

#endif
