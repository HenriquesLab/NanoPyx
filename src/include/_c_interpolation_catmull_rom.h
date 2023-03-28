#ifndef _C_INTERPOLATION_CATMULL_ROM_H
#define _C_INTERPOLATION_CATMULL_ROM_H

// Cubic function used in Catmull-Rom interpolation
// https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull.E2.80.93Rom_spline
float _c_cubic(float v);

// Catmull-Rom interpolation
float _c_interpolate(float* image, float r, float c, int rows, int cols);

#endif  // _C_INTERPOLATION_CATMULL_ROM_H
