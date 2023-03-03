#include "_c_calculate_distance_weight.h"

#include <math.h>

double _c_calculate_dw(double distance, double tSS) {
    return  pow((distance * exp((-distance * distance) / tSS)), 4);
}
