# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

# import libraries here
import numpy as np
from ...__liquid_engine__ import LiquidEngine

cimport numpy as np

cdef extern from "_c_template_advanced.h":
    void _c_template_advanced(float* image) nogil; # Only use this if using a C function

class Template(LiquidEngine):
    """
    Template to implement new methods using the Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "Template" # change to name of your method
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        unthreaded_=True, threaded_=False, threaded_static_=False, 
                        threaded_dynamic_=False, threaded_guided_=False, opencl_=False) # change implemented run types to True

    def run(self, image, run_type = None): 
        return self._run(image)
    
    def benchmark(self, image):
        return super().benchmark(image)
    
     # tag-start: _le_radiality.Radiality._run_unthreaded
    def _run_unthreaded(self, float[:,:,:] image):

        for i in range(image.shape[0]):
            pass
        return np.asarray(image)
        # tag-end

    # tag-copy:  _le_radiality.Radiality._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range(image.shape[0])", "prange(image.shape[0])")
