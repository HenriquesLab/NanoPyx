# NanoPyx Liquid Engine

A NanoPyx Liquid Engine part tries to use the same code for both OpenCL and Cython calls.

Here's how you create an engine part:

1. create a .pyx file, then equivalent files with the same name but ending in \_.c, \_.h and \_.cl
2. write the base c-code you want to use insider the .c and .h files
3. use the "// c2cl-function: function_name" nomenclature inside the .cl file to keep a fresh copy of the implemented c-functions, use the nanopyx-c2cl command to copy them over automatically
4. enjoy
