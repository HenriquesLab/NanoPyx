#ifndef _C_INTEGRAL_IMAGE_H
#define _C_INTEGRAL_IMAGE_H

#include <Python.h>
#include <stddef.h>

void _c_integral_image(
    float* padded,
    float* integral,
    int n_row,
    int n_col,
    int n_channels,
    int t_row,
    int t_col,
    float var_diff);

#endif  // _C_INTEGRAL_IMAGE_2D_H
