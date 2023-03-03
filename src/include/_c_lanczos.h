#ifndef LANCZOS_KERNEL_H
#define LANCZOS_KERNEL_H

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

double _c_lanczos_kernel(double v, int taps);

#ifdef __cplusplus
}
#endif

#endif /* LANCZOS_KERNEL_H */
