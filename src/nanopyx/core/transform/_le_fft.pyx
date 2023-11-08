# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift

import pyvkfft.opencl
from pyvkfft.opencl import VkFFTApp

from cython.parallel import parallel, prange

from .__interpolation_tools__ import check_image
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array

class FFT(LiquidEngine):
    """
    FFT on a set of 3D images using the NanoPyx Liquid Engine.
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "FFT"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                        opencl_=True, threaded_=True)
        

    def run(self, image, run_type=None):
        image = check_image(image)
        return self._run(image, run_type=run_type)

    def benchmark(self, image):
        image = check_image(image)
        return super().benchmark(image)

    def _run_opencl(self, image, device=None, mem_div=1):
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        c_image = image.astype(np.complex64)
        output_image = np.empty_like(c_image)

        total_memory = (c_image[0,:,:].nbytes*2)
        
        max_slices = int((dc.global_mem_size // total_memory)/mem_div)
        max_slices = self._check_max_slices(image, max_slices)
        self._check_max_buffer_size(c_image[0:max_slices, :, :].nbytes, dc, max_slices)

        input_cl = cl_array.to_device(cl_queue,c_image[0:max_slices,:,:])
        output_cl = cl_array.empty_like(input_cl)
        
        app = VkFFTApp(input_cl.shape, input_cl.dtype, queue=cl_queue, inplace=False, norm=1, ndim=2, axes=(1,2))
        cl_queue.finish()

        for i in range(0, image.shape[0], max_slices):
            if image.shape[0] - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = image.shape[0] - i

            app.fft(input_cl,output_cl)
            output_image[i:i+n_slices,:,:] = output_cl.get()[:n_slices,:,:]

            if i+n_slices<image.shape[0]:
                next_input = image[i+n_slices:i+2*n_slices,:,:]
                next_input_length = next_input.shape[0]
                if next_input_length<input_cl.shape[0]:
                    lengthofpad = input_cl.shape[0]-next_input_length
                    padded_input = np.append(next_input,np.zeros((lengthofpad,next_input.shape[1],next_input.shape[2]),dtype=np.complex64),axis=0)
                    input_cl.set(padded_input)
                else:
                    input_cl.set(next_input)

            cl_queue.finish()


        return fftshift(output_image)
            
    def _run_threaded(self, float[:,:,:] image):

        FS = fftshift(fft2(image))

        return np.array(FS)
