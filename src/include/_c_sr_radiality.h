#ifndef _C_SR_RADIALITY_H
#define _C_SR_RADIALITY_H

#include <math.h>

float _c_calculate_radiality_per_subpixel(int i, int j, float* imGx, float* imGy, float* xRingCoordinates, float* yRingCoordinates, int magnification, float ringRadius, int nRingCoordinates, int radialityPositivityConstraint, int h, int w);

float _c_calculate_dk(float x, float y, float xc, float yc, float vGx, float vGy, float GMag, float ringRadius);

#endif