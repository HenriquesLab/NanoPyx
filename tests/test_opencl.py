import numpy as np
import pyopencl as cl
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)

import nanopyx.opencl as npxcl
from nanopyx.opencl import _cl_mandelbrot_benchmark


@npxcl.opencl_available
def test_sum_images():
    a = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    b = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    c_np_result = a + b

    for platform in cl.get_platforms():
        for device in platform.get_devices():
            # Simple speed test
            ctx = cl.Context([device])
            queue = cl.CommandQueue(
                ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
            )

            a_cl = pycl_array.to_device(queue, a)
            b_cl = pycl_array.to_device(queue, b)
            c_cl = pycl_array.empty_like(
                a_cl
            )  # Create an empty pyopencl destination array

            program = cl.Program(
                ctx,
                """
            __kernel void sum(__global const float *a, __global const float *b, __global float *c)
            {
                const int i = get_global_id(0);
                const int j = get_global_id(1);
                const int rows = get_global_size(0);
                const int idx = i*rows + j;
                c[idx] = a[idx] + b[idx];
            }""",
            ).build()  # Create the OpenCL program

            exec_evt = program.sum(
                queue, a.shape, None, a_cl.data, b_cl.data, c_cl.data
            )  # Enqueue the program for execution and store the result in c_cl
            exec_evt.wait()
            elapsed = 1e-9 * (exec_evt.profile.end - exec_evt.profile.start)

            assert c_cl.get().all() == c_np_result.all()
            print("Execution time of test: %g s" % elapsed)


@npxcl.opencl_available
def test_mandelbrot_benchmark():
    _cl_mandelbrot_benchmark._cl_mandelbrot(1000, 1000, 4)
