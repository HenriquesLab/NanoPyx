#ifndef PATCH_DISTANCE_H
#define PATCH_DISTANCE_H

#include <math.h>

// Function to calculate patch distance for 2D patches
float _c_patch_distance(const float* p1, const float* p2, const float* w, int patch_size, int iglobal, int jglobal, int n_col, float var);

#endif  // PATCH_DISTANCE_H
