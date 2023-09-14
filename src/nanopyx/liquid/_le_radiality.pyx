# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport sqrt, pi, fabs, cos, sin
from ..__liquid_engine__ import LiquidEngine
from ..__opencl__ import cl, cl_array
from .__interpolation_tools__ import check_image

# TODO is this supposed to be absolute?
from nanopyx.liquid import CRShiftAndMagnify
from nanopyx.core.utils.timeit import timeit2

cdef extern from "_c_interpolation_catmull_rom.h":
    pass

cdef extern from "_c_sr_radiality.h":
    float _c_calculate_radiality_per_subpixel(int i, int j, float* imGx, float* imGy, float* xRingCoordinates, float* yRingCoordinates, int magnification, float ringRadius, int nRingCoordinates, int radialityPositivityConstraint, int h, int w) nogil

cdef extern from "_c_gradients.h":
    void _c_gradient_radiality(float* image, float* imGc, float* imGr, int rows,
                          int cols) nogil

# cdef float Gx_Gy_MAGNIFICATION = 2.0

class Radiality(LiquidEngine):
    """
    Radial gradient convergence using the NanoPyx Liquid Engine
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "Radiality"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing,
                        unthreaded_=False, threaded_=True, threaded_static_=True, 
                        threaded_dynamic_=True, threaded_guided_=True, opencl_=True)

        self._default_benchmarks = {'OpenCL': {"(['shape(100, 100, 100)', 'shape(100, 500, 500)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [62500000000000.0, 0.9875555419998818, 0.9819192090008073, 0.9873175410002659], "(['shape(100, 150, 150)', 'shape(100, 750, 750)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [316406250000000.0, 2.204043416999866, 2.202929583000696, 2.2014064169998164]}, 'Threaded': {"(['shape(100, 100, 100)', 'shape(100, 500, 500)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [62500000000000.0, 3.56841600000007, 3.5533971249997194, 3.2480534169999373], "(['shape(100, 150, 150)', 'shape(100, 750, 750)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [316406250000000.0, 7.231931541999984, 7.29744833300083, 7.1096272920003685]}, 'Threaded_dynamic': {"(['shape(100, 100, 100)', 'shape(100, 500, 500)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [62500000000000.0, 2.7894346250000126, 2.778901292000228, 2.714467834000061], "(['shape(100, 150, 150)', 'shape(100, 750, 750)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [316406250000000.0, 6.166010375000042, 6.16195045799941, 6.185223917000258]}, 'Threaded_guided': {"(['shape(100, 100, 100)', 'shape(100, 500, 500)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [62500000000000.0, 2.7807863329999236, 2.714370166999288, 2.720014749999791], "(['shape(100, 150, 150)', 'shape(100, 750, 750)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [316406250000000.0, 6.414935042000252, 6.137087832999896, 6.186802750000425]}, 'Threaded_static': {"(['shape(100, 100, 100)', 'shape(100, 500, 500)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [62500000000000.0, 3.3877954579998004, 3.466373499999463, 3.2974376660004054], "(['shape(100, 150, 150)', 'shape(100, 750, 750)', 'number(5)', 'number(0.5)', 'number(0)', True, True], {})": [316406250000000.0, 7.016077875000519, 7.207373291999829, 7.122106583999994]}}

    def run(self, image, image_interp, magnification: int = 5, ringRadius: float = 0.5, border: int = 0, radialityPositivityConstraint: bool = True, doIntensityWeighting: bool = True, run_type = None): 
        image = check_image(image)
        image_interp = check_image(image_interp)
        return self._run(image, image_interp, magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting, run_type=run_type)
    
    def benchmark(self, image, image_interp, magnification: int = 5, ringRadius: float = 0.5, border: int = 0, radialityPositivityConstraint: bool = True, doIntensityWeighting: bool = True): 
        image = check_image(image)
        image_interp = check_image(image_interp)
        return super().benchmark(image, image_interp, magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting)
    
     # tag-start: _le_radiality.Radiality._run_unthreaded
    def _run_unthreaded(self, float[:,:,:] image, float[:,:,:] image_interp, magnification: int = 5, ringRadius: float = 0.5, border: int = 0, radialityPositivityConstraint: bool = True, doIntensityWeighting: bool = True):

        cdef int _magnification = magnification
        cdef int _border = border
        cdef float _ringRadius = ringRadius * magnification
        cdef int _doIntensityWeighting = doIntensityWeighting
        cdef int _radialityPositivityConstraint = radialityPositivityConstraint
        cdef int nRingCoordinates = 12
        cdef float angleStep = (pi * 2.) / nRingCoordinates
        cdef float[12] xRingCoordinates, yRingCoordinates

        with nogil:
            for angleIter in range(nRingCoordinates):
                xRingCoordinates[angleIter] = cos(angleStep * angleIter) * _ringRadius
                yRingCoordinates[angleIter] = sin(angleStep * angleIter) * _ringRadius
        
        cdef int nFrames = image.shape[0]
        cdef int h = image.shape[1]
        cdef int w = image.shape[2]

        
        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, h*magnification, w*magnification), dtype=np.float32)

        cdef int f, j, i
        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], h, w)
                 for j in range((1 + _border) * _magnification, (h - 1 - _border) * _magnification):
                    for i in range((1 + _border) * _magnification, (w - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w) * image_interp[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w)

        return np.asarray(imRad)
        # tag-end

    # tag-copy:  _le_radiality.Radiality._run_unthreaded; replace("_run_unthreaded", "_run_threaded"); replace("range((1 + _border) * _magnification, (h - 1 - _border) * _magnification)", "prange((1 + _border) * _magnification, (h - 1 - _border) * _magnification)")
    def _run_threaded(self, float[:,:,:] image, float[:,:,:] image_interp, magnification: int = 5, ringRadius: float = 0.5, border: int = 0, radialityPositivityConstraint: bool = True, doIntensityWeighting: bool = True):

        cdef int _magnification = magnification
        cdef int _border = border
        cdef float _ringRadius = ringRadius * magnification
        cdef int _doIntensityWeighting = doIntensityWeighting
        cdef int _radialityPositivityConstraint = radialityPositivityConstraint
        cdef int nRingCoordinates = 12
        cdef float angleStep = (pi * 2.) / nRingCoordinates
        cdef float[12] xRingCoordinates, yRingCoordinates

        with nogil:
            for angleIter in range(nRingCoordinates):
                xRingCoordinates[angleIter] = cos(angleStep * angleIter) * _ringRadius
                yRingCoordinates[angleIter] = sin(angleStep * angleIter) * _ringRadius
        
        cdef int nFrames = image.shape[0]
        cdef int h = image.shape[1]
        cdef int w = image.shape[2]
        
        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, h*magnification, w*magnification), dtype=np.float32)

        cdef int f, j, i
        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], h, w)
                 for j in prange((1 + _border) * _magnification, (h - 1 - _border) * _magnification):
                    for i in range((1 + _border) * _magnification, (w - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w) * image_interp[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w)

        return np.asarray(imRad)
        # tag-end

    def _run_threaded_static(self, float[:,:,:] image, float[:,:,:] image_interp, magnification: int = 5, ringRadius: float = 0.5, border: int = 0, radialityPositivityConstraint: bool = True, doIntensityWeighting: bool = True):

        cdef int _magnification = magnification
        cdef int _border = border
        cdef float _ringRadius = ringRadius * magnification
        cdef int _doIntensityWeighting = doIntensityWeighting
        cdef int _radialityPositivityConstraint = radialityPositivityConstraint
        cdef int nRingCoordinates = 12
        cdef float angleStep = (pi * 2.) / nRingCoordinates
        cdef float[12] xRingCoordinates, yRingCoordinates

        with nogil:
            for angleIter in range(nRingCoordinates):
                xRingCoordinates[angleIter] = cos(angleStep * angleIter) * _ringRadius
                yRingCoordinates[angleIter] = sin(angleStep * angleIter) * _ringRadius
        
        cdef int nFrames = image.shape[0]
        cdef int h = image.shape[1]
        cdef int w = image.shape[2]
        
        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, h*magnification, w*magnification), dtype=np.float32)

        cdef int f, j, i
        with nogil:
            for f in range(nFrames):
                _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], h, w)
                for j in prange((1 + _border) * _magnification, (h - 1 - _border) * _magnification, schedule = "static"):
                    for i in range((1 + _border) * _magnification, (w - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w) * image_interp[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w)

        return np.asarray(imRad)

    def _run_threaded_dynamic(self, float[:,:,:] image, float[:,:,:] image_interp, magnification: int = 5, ringRadius: float = 0.5, border: int = 0, radialityPositivityConstraint: bool = True, doIntensityWeighting: bool = True):

        cdef int _magnification = magnification
        cdef int _border = border
        cdef float _ringRadius = ringRadius * magnification
        cdef int _doIntensityWeighting = doIntensityWeighting
        cdef int _radialityPositivityConstraint = radialityPositivityConstraint
        cdef int nRingCoordinates = 12
        cdef float angleStep = (pi * 2.) / nRingCoordinates
        cdef float[12] xRingCoordinates, yRingCoordinates

        with nogil:
            for angleIter in range(nRingCoordinates):
                xRingCoordinates[angleIter] = cos(angleStep * angleIter) * _ringRadius
                yRingCoordinates[angleIter] = sin(angleStep * angleIter) * _ringRadius
        
        cdef int nFrames = image.shape[0]
        cdef int h = image.shape[1]
        cdef int w = image.shape[2]

        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, h*magnification, w*magnification), dtype=np.float32)

        cdef int f, j, i
        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], h, w)
                 for j in prange((1 + _border) * _magnification, (h - 1 - _border) * _magnification, schedule = "dynamic"):
                    for i in range((1 + _border) * _magnification, (w - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w) * image_interp[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w)
        
        return np.asarray(imRad)

    def _run_threaded_guided(self, float[:,:,:] image, float[:,:,:] image_interp, magnification: int = 5, ringRadius: float = 0.5, border: int = 0, radialityPositivityConstraint: bool = True, doIntensityWeighting: bool = True):

        cdef int _magnification = magnification
        cdef int _border = border
        cdef float _ringRadius = ringRadius * magnification
        cdef int _doIntensityWeighting = doIntensityWeighting
        cdef int _radialityPositivityConstraint = radialityPositivityConstraint
        cdef int nRingCoordinates = 12
        cdef float angleStep = (pi * 2.) / nRingCoordinates
        cdef float[12] xRingCoordinates, yRingCoordinates

        with nogil:
            for angleIter in range(nRingCoordinates):
                xRingCoordinates[angleIter] = cos(angleStep * angleIter) * _ringRadius
                yRingCoordinates[angleIter] = sin(angleStep * angleIter) * _ringRadius
        
        cdef int nFrames = image.shape[0]
        cdef int h = image.shape[1]
        cdef int w = image.shape[2]
        
        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, h*magnification, w*magnification), dtype=np.float32)

        cdef int f, j, i
        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], h, w)
                 for j in prange((1 + _border) * _magnification, (h - 1 - _border) * _magnification, schedule = "guided"):
                    for i in range((1 + _border) * _magnification, (w - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w) * image_interp[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, h, w)

        return np.asarray(imRad)

    
    def _run_opencl(self, image, image_interp, magnification=5, ringRadius=0.5, border=0, radialityPositivityConstraint=True, doIntensityWeighting=True, device=None):

        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

        code = self._get_cl_code("_le_radiality.cl", device['DP'])

        cdef float _ringRadius = ringRadius * magnification
        
        cdef int nRingCoordinates = 12
        cdef float angleStep = (pi * 2.) / nRingCoordinates
        cdef float[12] xRingCoordinates, yRingCoordinates

        with nogil:
            for angleIter in range(nRingCoordinates):
                xRingCoordinates[angleIter] = cos(angleStep * angleIter) * _ringRadius
                yRingCoordinates[angleIter] = sin(angleStep * angleIter) * _ringRadius

    
        cdef int nFrames = image.shape[0]
        cdef int h = image.shape[1]
        cdef int w = image.shape[2]

        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float[:,:,:] image_MV = image
        with nogil:
            for f in range(nFrames):
                _c_gradient_radiality(&image_MV[f,0,0], &imGx[f,0,0], &imGy[f,0,0], h, w)

        image_in = cl_array.to_device(cl_queue, image)
        imageinter_in = cl_array.to_device(cl_queue, image_interp)
        imGx_in = cl_array.to_device(cl_queue, np.array(imGx, dtype=np.float32))
        imGy_in = cl_array.to_device(cl_queue, np.array(imGy, dtype=np.float32))
        imRad_out = cl_array.zeros(cl_queue, (nFrames, h*magnification, w*magnification), dtype=np.float32)

        xRingCoordinates_in = cl_array.to_device(cl_queue, np.array(xRingCoordinates, dtype=np.float32))
        yRingCoordinates_in = cl_array.to_device(cl_queue, np.array(yRingCoordinates, dtype=np.float32))

        # Grid size
        lowest_row = (1 + border) * magnification 
        highest_row = (h - 1 - border) * magnification

        lowest_col = (1 + border) * magnification
        highest_col = (w - 1 - border) * magnification

        prg = cl.Program(cl_ctx, code).build()  
 
        prg.radiality(
            cl_queue,
            (nFrames, highest_row - lowest_row, highest_col - lowest_col),
            None,
            image_in.data,
            imageinter_in.data,
            imGx_in.data,
            imGy_in.data,
            imRad_out.data,
            xRingCoordinates_in.data,
            yRingCoordinates_in.data,
            np.int32(magnification),
            np.float32(_ringRadius),
            np.int32(nRingCoordinates),
            np.int32(radialityPositivityConstraint),
            np.int32(border),
            np.int32(h),
            np.int32(w)
        )
    
        cl_queue.finish()        
        
        return np.asarray(imRad_out.get(),dtype=np.float32)
    