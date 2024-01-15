<%!
import sys
import os

sys.path.append(os.getcwd())
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

__kernel void nlm_denoising(__global float *padded,__global int *shifts, __global float *weights, __global float *integral,__global float *result,int n_row, int n_col, int patch_size, int patch_distance, float h2s2, float var) {

    int f = get_global_id(0);
    int nframes = get_global_size(0);
    int shift = get_global_id(1);

    int t_row = shifts[shift*2];
    int t_col = shifts[shift*2+1];

    int offset = patch_size / 2;
    int row_start = max(offset, offset-t_row);
    int row_end = min(n_row-offset, n_row-offset-t_row);

    float alpha = 1;
    if(t_col==0){
        alpha = 0.5;
    }

    _c_integral_image(&padded[f*n_row*n_col],&integral[shift*nframes*n_row*n_col+f*n_row*n_col],n_row,n_col,t_row,t_col,var);

    int row,col;
    int row_shift, col_shift;
    float distance, weight;
    for(row=row_start;row<row_end;row++){
        for(col=offset;col<n_col-offset-t_col;col++){
            row_shift = row+t_row;
            col_shift = col+t_col;
            distance = _c_integral_to_distance(&integral[shift*nframes*n_row*n_col+f*n_row*n_col],n_row,n_col,row,col,offset,h2s2);
            weight = alpha * exp(-distance);

            weights[shift*nframes*n_row*n_col+f*n_row*n_col+row*n_col+col] = weights[shift*nframes*n_row*n_col+f*n_row*n_col+row*n_col+col] + weight;
            weights[shift*nframes*n_row*n_col+f*n_row*n_col+row_shift*n_col+col_shift] = weights[shift*nframes*n_row*n_col+f*n_row*n_col+row_shift*n_col+col_shift] + weight;

            result[shift*nframes*n_row*n_col+f*n_row*n_col+row*n_col+col] = result[shift*nframes*n_row*n_col+f*n_row*n_col+row*n_col+col] + weight * padded[f*n_row*n_col+row_shift*n_col+col_shift];
            result[shift*nframes*n_row*n_col+f*n_row*n_col+row_shift*n_col+col_shift] = result[shift*nframes*n_row*n_col+f*n_row*n_col+row_shift*n_col+col_shift] + weight * padded[f*n_row*n_col+row*n_col+col];
        }
    }
}

__kernel void nlm_normalizer(__global float *weights, __global float *result, __global float *output, int pad_size, int n_shifts, int n_row, int n_col){

    int f = get_global_id(0);
    int nframes = get_global_size(0);
    int f_row = get_global_id(1) + pad_size;
    int f_col = get_global_id(2) + pad_size;

    float final_result = 0;
    float final_weight = 0;
    
    int shift;
    for(shift=0;shift<n_shifts;shift++){
        final_result = final_result + result[shift*nframes*n_row*n_col+f*n_row*n_col+f_row*n_col+f_col];
        final_weight = final_weight + weights[shift*nframes*n_row*n_col+f*n_row*n_col+f_row*n_col+f_col];
    }

    output[f*n_row*n_col+f_row*n_col+f_col] = final_result / final_weight;
    
}