# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport sqrt, pi, fabs, cos, sin
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array
from .__interpolation_tools__ import check_image

from ._le_interpolation_catmull_rom import ShiftAndMagnify as CRShiftAndMagnify

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

    
    def _run_opencl(self, image, image_interp, magnification=5, ringRadius=0.5, border=0, radialityPositivityConstraint=True, doIntensityWeighting=True, device=None, int mem_div=1):

        cl_ctx = cl.Context([device['device']])
        cl_queue = cl.CommandQueue(cl_ctx)

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

        cdef float[:,:,:] imGx = np.zeros(image.shape, dtype=np.float32) 
        cdef float[:,:,:] imGy = np.zeros(image.shape, dtype=np.float32)
        cdef float[:,:,:] image_MV = image
        with nogil:
            for f in range(nFrames):
                _c_gradient_radiality(&image_MV[f,0,0], &imGx[f,0,0], &imGy[f,0,0], h, w)

        image_out = np.zeros((nFrames, h*magnification, w*magnification), dtype=np.float32)

        x_ring_coords = np.asarray(xRingCoordinates)
        y_ring_coords = np.asarray(yRingCoordinates)

        # Calculate maximum number of slices that can fit in the GPU
        size_per_slice = 2*image[0,:,:].nbytes + image_interp[0,:,:].nbytes + imGx[0,:,:].nbytes + imGy[0,:,:].nbytes + x_ring_coords.nbytes + y_ring_coords.nbytes
        max_slices = int((device["device"].global_mem_size // (size_per_slice))/mem_div)
        max_slices = self._check_max_slices(image, max_slices)

        # Initialize Buffers
        mf = cl.mem_flags
        image_in = cl.Buffer(cl_ctx, mf.READ_ONLY, image[0:max_slices,:,:].nbytes)
        imageinter_in = cl.Buffer(cl_ctx, mf.READ_ONLY, image_interp[0:max_slices,:,:].nbytes)
        imGx_in = cl.Buffer(cl_ctx, mf.READ_ONLY, imGx[0:max_slices,:,:].nbytes)
        imGy_in = cl.Buffer(cl_ctx, mf.READ_ONLY, imGy[0:max_slices,:,:].nbytes)
        xRingCoordinates_in = cl.Buffer(cl_ctx, mf.READ_ONLY, x_ring_coords.nbytes)
        yRingCoordinates_in = cl.Buffer(cl_ctx, mf.READ_ONLY, y_ring_coords.nbytes)
        imRad_out = cl.Buffer(cl_ctx, mf.WRITE_ONLY, image_out[0:max_slices,:,:].nbytes)

        cl.enqueue_copy(cl_queue, image_in, image[0:max_slices,:,:]).wait()
        cl.enqueue_copy(cl_queue, imageinter_in, image_interp[0:max_slices,:,:]).wait()
        cl.enqueue_copy(cl_queue, imGx_in, imGx[0:max_slices,:,:]).wait()
        cl.enqueue_copy(cl_queue, imGy_in, imGy[0:max_slices,:,:]).wait()
        cl.enqueue_copy(cl_queue, xRingCoordinates_in, x_ring_coords).wait()
        cl.enqueue_copy(cl_queue, yRingCoordinates_in, y_ring_coords).wait()

        # Grid size
        lowest_row = (1 + border) * magnification 
        highest_row = (h - 1 - border) * magnification

        lowest_col = (1 + border) * magnification
        highest_col = (w - 1 - border) * magnification

        code = self._get_cl_code("_le_radiality.cl", device['DP'])
        prg = cl.Program(cl_ctx, code).build()
        knl = prg.radiality

        for i in range(0, nFrames, max_slices):
            if nFrames - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = nFrames - i

            knl(
                cl_queue,
                (n_slices, highest_row - lowest_row, highest_col - lowest_col),
                self.get_work_group(device['device'],(n_slices, highest_row - lowest_row, highest_col - lowest_col)),
                image_in,
                imageinter_in,
                imGx_in,
                imGy_in,
                imRad_out,
                xRingCoordinates_in,
                yRingCoordinates_in,
                np.int32(magnification),
                np.float32(_ringRadius),
                np.int32(nRingCoordinates),
                np.int32(radialityPositivityConstraint),
                np.int32(border),
                np.int32(h),
                np.int32(w)
            )

            cl.enqueue_copy(cl_queue, image_out[i:i+n_slices,:,:], imRad_out).wait()

            if i<=nFrames-max_slices:
                cl.enqueue_copy(cl_queue, image_in, image[i+n_slices:i+2*n_slices,:,:]).wait()
                cl.enqueue_copy(cl_queue, imageinter_in, image_interp[i+n_slices:i+2*n_slices,:,:]).wait()
                cl.enqueue_copy(cl_queue, imGx_in, imGx[i+n_slices:i+2*n_slices,:,:]).wait()
                cl.enqueue_copy(cl_queue, imGy_in, imGy[i+n_slices:i+2*n_slices,:,:]).wait()
    
            cl_queue.finish()        
        
        image_in.release()
        imageinter_in.release()
        imGx_in.release()
        imGy_in.release()
        xRingCoordinates_in.release()
        yRingCoordinates_in.release()
        imRad_out.release()

        return image_out
    