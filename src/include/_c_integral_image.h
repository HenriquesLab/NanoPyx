#ifndef _C_INTEGRAL_IMAGE_H
#define _C_INTEGRAL_IMAGE_H

void _c_integral_image(
    float* padded,
    float* integral,
    int n_row,
    int n_col,
    int t_row,
    int t_col,
    float var_diff);

#endif  // _C_INTEGRAL_IMAGE_H
