import pyopencl as cl
from pyopencl import array as cl_array
import numpy as np

def arrest_gpu(dc):
    
    # QUEUE AND CONTEXT
    cl_ctx = cl.Context([dc])
    cl_queue = cl.CommandQueue(cl_ctx)

    kernel_code = """
                __kernel void stress_test(__global float* output){
                int gid = get_global_id(0);
                float result = 0.0f;
                for (int i = 0; i < 600000; ++i){
                    result += sin((float)(gid+i)) * cos((float)(gid-i));
                }
                output[gid] = result;
                }
                """

    prg = cl.Program(cl_ctx, kernel_code).build()

    num_elements = dc.max_work_group_size * 7500000
    gpu_output = cl.Buffer(cl_ctx, cl.mem_flags.WRITE_ONLY, size=num_elements*4)

    try:
        print("Start")
        while True:
            global_size = (num_elements,)
            local_size = None
            prg.stress_test(cl_queue, global_size, local_size, gpu_output)
            cl_queue.finish()
            #cpu_output = np.empty(num_elements, dtype=np.float32)
            #cl.enqueue_copy(cl_queue, cpu_output, gpu_output)
    except KeyboardInterrupt:
        print("Interrupt")


if __name__ == "__main__":

    devices = []
    for platform in cl.get_platforms():
        if 'Microsoft' in platform.vendor or 'Oclgrind' in platform.vendor or 'Intel' in platform.vendor: # Force Nvidia GPU
            continue
        for dev in platform.get_devices():
            # check if the device is a GPU
            if "GPU" not in cl.device_type.to_string(dev.type):
                continue
            if 'cl_khr_fp64' in dev.extensions.strip().split(' '):
                cl_dp = True
            else:
                cl_dp = False

            devices.append({'device':dev, 'DP':cl_dp})
    
    print(devices[0])
    arrest_gpu(devices[0]['device'])