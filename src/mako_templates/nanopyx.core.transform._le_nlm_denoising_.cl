float _c_patch_distance(const __global float *p1, const __global float *p2, const __global float *w, int patch_size, int iglobal, int jglobal, int n_col, float var);


float _c_patch_distance(const __global float *p1, const __global float *p2, const __global float *w, int patch_size, int iglobal, int jglobal, int n_col, float var) {
    int i, j;
    const float DISTANCE_CUTOFF = 5.0;
    float tmp_diff = 0.0;
    float distance = 0.0;

    for (i = 0; i < patch_size; i++) {
        // exp of large negative numbers will be 0, so we'd better stop
        if (distance > DISTANCE_CUTOFF) {
            printf("Distance exceeds cutoff. Returning 0.0.\n");
            return 0.0;
        }
        for (j = 0; j < patch_size; j++) {
            tmp_diff = p1[i * n_col + j] - p2[(iglobal+i) * n_col + (jglobal+j)];
            distance = distance + w[i * patch_size + j] * (tmp_diff * tmp_diff - var);
            // printf("i=%d, j=%d, tmp_diff=%f, w=%f, distance=%f\n", i, j, tmp_diff, w[i * patch_size + j], distance);
            
        }
    }
    // printf("distance=%f\n", exp(-max(0.0, distance)));
    return exp(-(max(0.0, distance)));
}

__kernel void
nlm_denoising(__global float *padded_opencl, __global float *w_opencl, __global float *result_opencl, int n_row, int n_col, int patch_size, int patch_distance, int offset, float var) {

    int f = get_global_id(0);
    int row = get_global_id(1);
    int col = get_global_id(2);

    int n_row_padded = n_row + 2*offset;
    int n_col_padded = n_col + 2*offset;

    int i_start = row - min(patch_distance, row);
    int i_end = row + min(patch_distance + 1, n_row - row);
    int j_start = col - min(patch_distance, col);
    int j_end = col + min(patch_distance + 1, n_col - col);

    float new_value = 0.0;
    float weight_sum = 0.0;

    //printf("f=%i, n_row=%i, n_col=%i, patch_size=%i, patch_distance=%i, offset=%i, var=%f, n_row_padded=%i, n_col_padded=%i, central_value=%f, padded_value=%f\n", f, n_row, n_col, patch_size, patch_distance, offset, var, n_row_padded, n_col_padded, padded_opencl[74*148 + 74], padded_opencl[f * n_row_padded * n_col_padded + (74) * n_col_padded + (74)]);

    for (int i = i_start; i < i_end; i++) {
        for (int j = j_start; j < j_end; j++) {
            float weight = _c_patch_distance(
                            &padded_opencl[f * n_row_padded * n_col_padded + row * n_col_padded + col],
                            &padded_opencl[f * n_row_padded * n_col_padded],
                            &w_opencl[0], patch_size,
                            i, j, n_col + 2 * offset, var);

            weight_sum = (weight_sum + weight);

            new_value = (new_value + weight * padded_opencl[f * n_row_padded * n_col_padded + (i+offset) * n_col_padded + (j+offset)]);
        }
    }
    result_opencl[f * n_row * n_col + row * n_col + col] = new_value / weight_sum;
}