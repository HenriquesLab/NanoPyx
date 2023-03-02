#ifndef LANCZOS_KERNEL_H
#define LANCZOS_KERNEL_H

#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

    double _c_lanczos_kernel(double v, int taps);

#ifdef __cplusplus
}
#endif

#endif /* LANCZOS_KERNEL_H */
