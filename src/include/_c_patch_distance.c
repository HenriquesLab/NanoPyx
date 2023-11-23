#include <math.h>

float _c_patch_distance(const float* p1, const float* p2, const float* w, int patch_size, int n_col, float var) {
    int i, j;
    const float DISTANCE_CUTOFF = 5.0;
    float tmp_diff, distance = 0;

    for (i = 0; i < patch_size; i++) {
        // exp of large negative numbers will be 0, so we'd better stop
        if (distance > DISTANCE_CUTOFF) {
            return 0.0;
        }
        for (j = 0; j < patch_size; j++) {
            tmp_diff = p1[i * patch_size + j] - p2[i * n_col + j];
            distance += w[i * patch_size + j] * (tmp_diff * tmp_diff - var);
        }
    }

    return expf(-fmaxf(0.0, distance));
}