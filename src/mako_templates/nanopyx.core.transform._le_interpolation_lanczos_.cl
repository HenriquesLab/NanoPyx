<%!
import sys
import os

sys.path.append(os.getcwd())
from src.scripts.c2cl import extract_batch_code

c_function_names = [('_c_interpolation_lanczos.c','_c_lanczos_kernel'),('_c_interpolation_lanczos.c','_c_interpolate')]

headers, functions = extract_batch_code(c_function_names)

defines = [('TAPS',4),('HALF_TAPS',2),('M_PI','3.14159265359f')]

%>

<%inherit file="_le_interpolation_base.cl"/>

