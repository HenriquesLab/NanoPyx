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
nlm_denoising(__global float *padded, __global float *result, __global float* integral, __global float* M, __global float* Z, const int f, const int n_row, const int n_col, const int offset, const float var, const float h2s2) {
        int t_row = get_global_id(0);
        int t_col = get_global_id(1);

        int size_col = get_global_size(1);

        int n = t_row * size_col + t_col;

        float  alpha = 1;
        if (t_col == 0){
            alpha = 0.5;
        }
    
        _c_integral_image(&padded[f*n_col*n_rowes], &integral[n*n_col*n_row], n_row, n_col, t_row, t_col, var);
        
        int row_start = max(offset,offset-t_row);
        int row_end = min(n_row-offset, n_row-offset-t_row);
        int col_start = max(offset,offset-t_col);
        int col_end = min(n_col-offset, n_col-offset-t_col);


        int row, col;
        float distance, weight;
        for (row=row_start;row<row_end;row++){
            for (col=col_start;col<col_end;col++){
                distance = _c_integral_to_distance(&integral[n*n_col*n_row], n_row, n_col, row, col, offset, h2s2);
                if (distance <= 5.0){
                    weight = alpha * exp(-distance);
                    result[f*n_row*n_col + row * n_col + col] = result[f*n_row*n_col + row * n_col + col] + weight*result[f*n_row*n_col + (row+t_row) * n_col + (col+t_col)];
                    M[row * n_col + col] = max(M[row * n_col + col],weight);
                    Z[row * n_col + col] = Z[row * n_col + col] + weight;
                }
            }
        }
}

__kernel void
nlm_normalizer(__global float *padded, __global float *result, __global float* M, __global float* Z, const int f) {

        int row = get_global_id(0);
        int col = get_global_id(1);

        int n_row = get_global_size(0);
        int n_col = get_global_size(1);

        result[f*n_row*n_col + row * n_col + col] = result[f*n_row*n_col + row * n_col + col] + M[row * n_col + col] * padded[f*n_row*n_col + row * n_col + col];
        result[f*n_row*n_col + row * n_col + col] = result[f*n_row*n_col + row * n_col + col] / (Z[row * n_col + col]+M[row * n_col + col]);
}

