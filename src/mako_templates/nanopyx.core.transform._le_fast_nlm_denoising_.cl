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
nlm_denoising(__global float *padded, __global float *result, __global float* integral, __global float* Z, const int f, const int n_row, const int n_col, const int offset, const float var, const float h2s2, const int patch_distance) {
        int t_row = get_global_id(0)-patch_distance;
        int t_col = get_global_id(1);

        int size_col = get_global_size(1);
        int current_patch = get_global_id(0)*size_col*n_row*n_col+t_col*n_row*n_col;

        float  alpha = 1;
        if (t_col == 0){
            alpha = 0.5;
        }
    
        _c_integral_image(&padded[f*n_col*n_row], &integral[current_patch], n_row, n_col, t_row, t_col, var);
        
        int row_start = max(offset,offset-t_row);
        int row_end = min(n_row-offset, n_row-offset-t_row);
        
        int col_start = offset;
        int col_end = n_col-offset-t_col;

        int row_shift, col_shift;

        int row, col;
        float distance, weight;
        for (row=row_start;row<row_end;row++){
            row_shift = row+t_row;
            for (col=col_start;col<col_end;col++){
                distance = _c_integral_to_distance(&integral[current_patch], n_row, n_col, row, col, offset, h2s2);
                if (distance <= 5.0){
                    col_shift = col+t_col;
                    weight = alpha * exp(-distance);
                    Z[f*n_col*n_row+row*n_col+col] = Z[f*n_col*n_row+row*n_col+col] + weight;
                    Z[f*n_col*n_row+row_shift*n_col+col_shift]  = Z[f*n_col*n_row+row_shift*n_col+col_shift] + weight;
                    
                    result[f*n_col*n_row+row*n_col+col] = result[f*n_col*n_row+row*n_col+col] + weight * padded[f*n_col*n_row+row_shift*n_col+col_shift];
                    result[f*n_col*n_row+row_shift*n_col+col_shift]  = result[f*n_col*n_row+row_shift*n_col+col_shift] +  weight * padded[f*n_col*n_row+row*n_col+col];
                }
            }
        }
}

__kernel void
nlm_normalizer(__global float *result, __global float* Z, const int f, const int pad_size) {

        int row = get_global_id(0)+pad_size;
        int col = get_global_id(1)+pad_size;

        int n_row = get_global_size(0)+pad_size*2;
        int n_col = get_global_size(1)+pad_size*2;

        result[f*n_row*n_col + row * n_col + col] = result[f*n_row*n_col + row * n_col + col] / Z[f*n_row*n_col + row * n_col + col];
        
}
