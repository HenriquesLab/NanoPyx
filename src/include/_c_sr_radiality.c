#include <math.h>
#include "_c_interpolation_catmull_rom.h"

float _c_calculate_dk(float x, float y, float xc, float yc, float vGx, float vGy, float GMag, float ringRadius){
    float Dk = 0;
    if (GMag != 0) {
        Dk = 1 - (fabs(vGy * (xc - x) - vGx * (yc - y)) / GMag) / ringRadius;
        Dk = Dk * Dk;
    } 
    return Dk;
}
