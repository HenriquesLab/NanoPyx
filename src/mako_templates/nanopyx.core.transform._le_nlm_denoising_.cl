<%!
from src.scripts.c2cl import extract_batch_code

c_function_names = [('_c_integral_image.c','_c_integral_image'),('_c_integral_to_distance.c','_c_integral_to_distance')]

headers, functions = extract_batch_code(c_function_names)

defines = []

%>

% for h in self.attr.headers:
${h}
% endfor

% for d in self.attr.defines:
#define ${d[0]} ${d[1]}
% endfor

% for f in self.attr.functions:
${f}

% endfor


__kernel void
nlm_denoising(__global float *padded_img, __global float *result, __global float *weights, __global float *integral, const int f,
              const int n_row, const int n_col, const int patch_distance, const float var, const int offset, const float h2s2) {

   
    int patch_row = get_global_id(0);
    int t_row = patch_row - patch_distance;
    int row_start = max(offset, offset - t_row);
    int row_end = min(n_row - offset, n_row - offset - t_row);

    int t_col;
    for (t_col=0;t_col<patch_distance+1;++t_col){

        float alpha;
        if (t_col==0){
            alpha = 0.5;
        } else {
            alpha = 1;
        }

        _c_integral_image(&padded_img[f*n_row*n_col], &integral[patch_row*n_row*n_col], n_row, n_col, t_row, t_col, var);

        int row, col, col_shift, row_shift;
        float weight, distance;
        
        for (row = row_start; row < row_end; ++row){
            row_shift = row + t_row;
            for (col=offset; col < n_col - offset - t_col;++col){
                distance = _c_integral_to_distance(&integral[patch_row*n_row*n_col], n_row, n_col, row, col, offset, h2s2);
                if (distance<=5.0){
                    col_shift = col+t_col;
                    weight = alpha * exp(-distance);
                    weights[row*n_col+col] += weight;
                    weights[row_shift*n_col+col_shift] += weight;

                    result[f*n_row*n_col+row*n_col+col] += weight * padded_img[f*n_row*n_col+row_shift*n_col+col_shift];
                    result[f*n_row*n_col+row_shift*n_col+col_shift] += weight * padded_img[f*n_row*n_col+row*n_col+col];
                }
            }
        }
    }

    // int pad_size = offset+patch_distance+1;
    // int row,col;
    // for (row=pad_size; row<n_row-pad_size;++row){
    //     for (col=pad_size; col<n_col-pad_size;++col){
    //         result[f*n_row*n_col+row*n_col+col] /= weights[row*n_col+col];
    //     }
    // }

}
    
