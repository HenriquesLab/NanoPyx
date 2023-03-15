#ifndef _C_INTERPOLATION_CATMULL_ROM_H
#define _C_INTERPOLATION_CATMULL_ROM_H

// Cubic function used in Catmull-Rom interpolation
// https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Catmull.E2.80.93Rom_spline
double _c_cubic(double v);

// Catmull-Rom interpolation
double _c_interpolate(float* image, double r, double c, int rows, int cols);

#endif  // _C_INTERPOLATION_CATMULL_ROM_H
