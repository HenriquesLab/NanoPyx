#include <math.h>
#include "_c_interpolation_catmull_rom.h"
#include "_c_gradients.h"
#include <stdio.h>

float _c_calculate_dk(float x, float y, float xc, float yc, float vGx, float vGy, float GMag, float ringRadius){
    float Dk = 0;
    if (GMag != 0) {
        Dk = 1 - (fabs(vGy * (xc - x) - vGx * (yc - y)) / GMag) / ringRadius;
        Dk = Dk * Dk;
    } 
    return Dk;
}

float _c_calculate_radiality_per_subpixel(int i, int j, float* imGx, float* imGy, float* xRingCoordinates, float* yRingCoordinates, int magnification, float ringRadius, int nRingCoordinates, int radialityPositivityConstraint, int h, int w) {
    int sampleIter;
    float x0, y0, xc, yc, xRing, yRing, vGx, vGy, GMag, Dk, DivDFactor = 0, CGH = 0;

    xc = i + 0.5;
    yc = j + 0.5;
    
    for (sampleIter = 0; sampleIter < nRingCoordinates; sampleIter++) {
        xRing = xRingCoordinates[sampleIter];
        yRing = yRingCoordinates[sampleIter];

        x0 = xc + xRing;
        y0 = yc + yRing;

        vGx = _c_interpolate(imGx, y0 / magnification, x0 / magnification, h, w);
        vGy = _c_interpolate(imGy, y0 / magnification, x0 / magnification, h, w);
        GMag = sqrt(vGx * vGx + vGy * vGy);

        Dk = _c_calculate_dk(x0, y0, xc, yc, vGx, vGy, GMag, ringRadius);

        if ((vGx * xRing + vGy * yRing) > 0) {
            DivDFactor -= Dk;
        } else {
            DivDFactor += Dk;
        }
    }

    DivDFactor /= nRingCoordinates;

    if (radialityPositivityConstraint == 1) {
        CGH = fmaxf(DivDFactor, 0);
    } else {
        CGH = DivDFactor;
    }

    return CGH;
}
