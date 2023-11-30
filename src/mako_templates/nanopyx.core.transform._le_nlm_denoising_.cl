<%!
from src.scripts.c2cl import extract_batch_code

c_function_names = [('_c_patch_distance.c','_c_patch_distance'),]

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
nlm_denoising(__global float *padded, __global float *w, __global float *result, const int n_row, const int n_col, const int patch_size, const int patch_distance, const int offset, const float var) {

    // kernel from the other implementation needs changing
    int f = get_global_id(0)
    int r = get_global_id(1)
    int c = get_global_id(2)

    int rows = get_global_size(1)
    int cols = get_global_size(2)

    int n_pixels = rows * cols

    
}