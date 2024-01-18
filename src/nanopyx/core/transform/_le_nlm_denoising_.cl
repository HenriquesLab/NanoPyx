__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void nlm_denoising(__read_only image2d_t padded, __write_only image2d_t result, __read_only image2d_t w, int patch_distance, int patch_size, int offset, float var) {

    int row = get_global_id(0);
    int col = get_global_id(1);

    int n_row = (int)get_global_size(0);
    int n_col = (int)get_global_size(1);

    int i_start = row - min(patch_distance, row);
    int i_end = row + min(patch_distance+1, n_row-row);
    int j_start = col - min(patch_distance, col);
    int j_end = col + min(patch_distance + 1, n_col - col);

    float weight_sum = 0;
    float weight;

    float new_value = 0;

    float distance;
    float tmp_diff;

    for (int i=i_start;i<i_end;i++) {
        for (int j=j_start;j<j_end;j++) {
            distance = 0;
            tmp_diff = 0;
            for (int is=0;is<patch_size;is++) {
                for (int js=0;js<patch_size;js++) {
                    tmp_diff = read_imagef(padded,sampler,(int2)(row+is,col+js)).x - read_imagef(padded,sampler,(int2)(i+is,j+js)).x;
                    distance = distance + read_imagef(w,sampler,(int2)(is,js)).x * (tmp_diff*tmp_diff-var);
                }
            }
            weight = exp(-max((float)(0.0),distance));
            weight_sum = weight_sum + weight;

            new_value = new_value + weight * read_imagef(padded,sampler,(int2)(i+offset,j+offset)).x;

        }
    }
    write_imagef(result,(int2)(row,col),new_value/weight_sum);
}


