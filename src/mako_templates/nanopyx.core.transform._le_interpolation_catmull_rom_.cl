<%!
from src.scripts.c2cl import extract_batch_code

c_function_names = [('_c_interpolation_catmull_rom.c','_c_interpolate')]

headers, functions = extract_batch_code(c_function_names)

defines = []

%>

<%inherit file="_le_interpolation_base.cl"/>

