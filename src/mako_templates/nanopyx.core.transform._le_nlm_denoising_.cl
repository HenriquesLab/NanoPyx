float _c_patch_distance(const __global float *p1, const __global float *p2, const __global float *w, int patch_size, int iglobal, int jglobal, int n_col, float var);


float _c_patch_distance(const __global float *p1, const __global float *p2, const __global float *w, int patch_size, int iglobal, int jglobal, int n_col, float var) {
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

    return exp(-max(0.0, distance));
}

__kernel void
nlm_denoising(__global float *padded, __global float *w, __global float *result, const int n_frames, const int n_row, const int n_col, const int patch_size, const int patch_distance, const int offset, const float var) {

    for (int f = 0; f < n_frames; ++f) {
        for (int row = 0; row < n_row; ++row) {
            int i_start = max(row - min(patch_distance, row), 0);
            int i_end = min(row + min(patch_distance + 1, n_row - row), n_row);

            for (int col = 0; col < n_col; ++col) {
                float new_value = 0.0;
                float weight_sum = 0.0;

                int j_start = max(col - min(patch_distance, col), 0);
                int j_end = min(col + min(patch_distance + 1, n_col - col), n_col);

                for (int i = i_start; i < i_end; ++i) {
                    for (int j = j_start; j < j_end; ++j) {
                        int index_padded = f * n_row * n_col + i * n_col + j;
                        int index_w = i * patch_size + j;

                        float distance = _c_patch_distance(&padded[index_padded],
                                                            &padded[f * n_row * n_col],
                                                            &w[index_w], patch_size,
                                                            i, j, n_col + 2 * offset, var);

                        // Replace expf and fmaxf with OpenCL equivalents
                        float weight = exp(-max(0.0, distance));

                        weight_sum += weight;
                        new_value += weight * padded[index_padded];
                    }
                }

                if (weight_sum > 0.0) {
                    int index_result = f * n_row * n_col + row * n_col + col;
                    result[index_result] = new_value / weight_sum;
                }
            }
        }
    }
}