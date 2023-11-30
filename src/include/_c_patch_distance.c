#include <math.h>
#include <stdio.h>

float _c_patch_distance(const float* p1, const float* p2, const float* w, int patch_size, int iglobal, int jglobal, int n_col, float var) {
    int i, j;
    const float DISTANCE_CUTOFF = 5.0;
    float tmp_diff = 0.0;
    float distance = 0.0;

    for (i = 0; i < patch_size; i++) {
        // exp of large negative numbers will be 0, so we'd better stop
        if (distance > DISTANCE_CUTOFF) {
            // printf("Distance exceeds cutoff. Returning 0.0.\n");
            return 0.0;
        }
        for (j = 0; j < patch_size; j++) {
            tmp_diff = p1[i * n_col + j] - p2[(iglobal+i) * n_col + (jglobal+j)];
            distance = distance + w[i * patch_size + j] * (tmp_diff * tmp_diff - var);
            // printf("i=%d, j=%d, tmp_diff=%f, w=%f, distance=%f\n", i, j, tmp_diff, w[i * patch_size + j], distance);
        }
    }

    return expf(-fmaxf(0.0, distance));
}