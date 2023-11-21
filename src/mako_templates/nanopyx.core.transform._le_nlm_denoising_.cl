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

