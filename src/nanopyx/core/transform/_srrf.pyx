# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import os
from pathlib import Path
import inspect

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange

from libc.math cimport sqrt, pi, fabs, cos, sin
from ...__opencl__ import cl, cl_array

cdef extern from "_c_interpolation_catmull_rom.h":
    float _c_interpolate(float *image, float row, float col, int rows, int cols) nogil

cdef extern from "_c_sr_radiality.h":
    float _c_calculate_radiality_per_subpixel(int i, int j, float* imGx, float* imGy, float* xRingCoordinates, float* yRingCoordinates, int magnification, float ringRadius, int nRingCoordinates, int radialityPositivityConstraint, int h, int w) nogil

cdef extern from "_c_gradients.h":
    void _c_gradient_radiality(float* image, float* imGc, float* imGr, int rows, int cols) nogil

class SRRF():
    """
    SRRF using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self):
        
        self._designation = "SRRF_ST"
        self.mem_div = 1


    def get_work_group(self, device, shape):
        """
        Calculates work group size for a given device and shape of global work space
        """

        max_wg_dims = device.max_work_item_sizes[0:3]
        max_glo_dims = device.max_work_group_size

        three = self.get_highest_divisor(shape[2], max_wg_dims[2])
        max_two = max_glo_dims / three
        two = self.get_highest_divisor(shape[1], max_two)
        one = 1
        return (one, two, three)
    
    def _check_max_slices(self, input, number_of_max_slices):
        """
        Checks if number of maximum slices is greater than 0
        """
        if number_of_max_slices < 1:
            raise ValueError("This device doesn't have enough memory to run this function with this input")
        elif input.shape[0] < number_of_max_slices:
            return input.shape[0]
        else:
            return number_of_max_slices

    def _check_max_buffer_size(self, size, device, n_slices):
        """
        Checks if buffer size is larger than device maximum memory allocation size and n_slices is 1 and raises appropriate errors that are handled in the _run function.
        """
        if size > device.max_mem_alloc_size and n_slices == 1:
            raise ValueError("This device cannot handle this input size with these parameters, try using a smaller input or other parameters")

        if size > device.max_mem_alloc_size:
            raise cl.Error("Buffer size is larger than device maximum memory allocation size")

        return size

    def get_highest_divisor(self, size_, max_):
        """
        Returns the highest divisor of size_ that is still lower than max_
        """
        value = 1
        for i in range(1, int(np.sqrt(size_) + 1)):
            if size_ % i == 0:
                if i * i != size_:
                    div2 = size_ / i

                    if i < max_:
                        value = max(value, i)
                    if div2 < max_:
                        value = max(value, div2)
        return int(value)


    def _get_cl_code(self, file_name, cl_dp):
        """
        Retrieves the OpenCL code from the corresponding .cl file
        """
        cl_file = os.path.splitext(file_name)[0] + ".cl"

        if not os.path.exists(cl_file):
            cl_file = Path(os.path.abspath(inspect.getfile(self.__class__))).parent / file_name

        assert os.path.exists(cl_file), "Could not find OpenCL file: " + str(cl_file)

        kernel_str = open(cl_file).read()

        if not cl_dp:
            kernel_str = kernel_str.replace("double", "float")

        return kernel_str

    def run_opencl(self, image, magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting, device):

        mem_div = self.mem_div
        try:
            if self.mem_div > 999:
                raise ValueError(f"Maximum memory division factor achieved")
            result = self._run_bare_opencl(image, magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting, device, mem_div)

        except (cl.MemoryError, cl.LogicError) as e:
            print("Found: ", e)
            print("Reducing maximum buffer size and trying again...")
            self.mem_div += 1
            mem_div = self.mem_div
            result = self.run_opencl(image, magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting, device)
        except cl.Error as e:
            if e.__str__() == "Buffer size is larger than device maximum memory allocation size":
                print("Found: ", e)
                print("Reducing maximum buffer size and trying again...")
                self.mem_div += 1
                mem_div = self.mem_div
                result = self.run_opencl(image, magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting, device)
            else:
                print(f"Unexpected error while trying to run")
                print(e)
                print("Please try again with another run type")
                result = None
        except Exception as e:
            print(f"Unexpected error while trying to run")
            print(e)
            print("Please try again with another run type")
            result = None

        self.mem_div = 1
        return result


    def _run_bare_opencl(self, image, magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting, device, mem_div):

        # TODO doIntensityWeighting is irrelevant on gpu2
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
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
        image_interp = np.zeros((nFrames, h*magnification, w*magnification))

        x_ring_coords = np.asarray(xRingCoordinates, dtype=np.float32)
        y_ring_coords = np.asarray(yRingCoordinates, dtype=np.float32)

        # Calculate maximum number of slices that can fit in the GPU
        size_per_slice = 2*image[0,:,:].nbytes + image_interp[0,:,:].nbytes + imGx[0,:,:].nbytes + imGy[0,:,:].nbytes
        max_slices = int(((x_ring_coords.nbytes + y_ring_coords.nbytes)+(device["device"].global_mem_size // size_per_slice))/mem_div)
        max_slices = self._check_max_slices(image, max_slices)

        # Initialize Buffers
        mf = cl.mem_flags
        image_in = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(image[0:max_slices,:,:].nbytes, device['device'], max_slices))
        imageinter_in = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(image_interp[0:max_slices,:,:].nbytes, device['device'], max_slices))
        imGx_in = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(imGx[0:max_slices,:,:].nbytes, device['device'], max_slices))
        imGy_in = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(imGy[0:max_slices,:,:].nbytes, device['device'], max_slices))
        xRingCoordinates_in = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(x_ring_coords.nbytes, device['device'], max_slices))
        yRingCoordinates_in = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(y_ring_coords.nbytes, device['device'], max_slices))
        imRad_out = cl.Buffer(cl_ctx, mf.WRITE_ONLY, self._check_max_buffer_size(image_out[0:max_slices,:,:].nbytes, device['device'], max_slices))

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

        cr_code = self._get_cl_code("_le_interpolation_catmull_rom_.cl", device['DP'])
        cr_prg = cl.Program(cl_ctx, cr_code).build(options=["-cl-mad-enable -cl-fast-relaxed-math"])
        cr_knl = cr_prg.shiftAndMagnify

        rad_code = self._get_cl_code("_le_radiality.cl", device['DP'])
        rad_prg = cl.Program(cl_ctx, rad_code).build(options=["-cl-mad-enable -cl-fast-relaxed-math"])
        rad_knl = rad_prg.radiality

        for i in range(0, nFrames, max_slices):
            if nFrames - i >= max_slices:
                n_slices = max_slices
            else:
                n_slices = nFrames - i

            cr_knl(cl_queue,
                (n_slices, int(image.shape[1]*magnification), int(image.shape[2]*magnification)),
                self.get_work_group(dc, (n_slices, image.shape[1]*magnification, image.shape[2]*magnification)), 
                image_in,
                imageinter_in,
                np.float32(0),
                np.float32(0),
                np.float32(magnification),
                np.float32(magnification)).wait()

            rad_knl(
                cl_queue,
                (n_slices, highest_row - lowest_row, highest_col - lowest_col),
                self.get_work_group(dc,(n_slices, highest_row - lowest_row, highest_col - lowest_col)),
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
            ).wait()

            cl.enqueue_copy(cl_queue, image_out[i:i+n_slices,:,:], imRad_out).wait()

            if i+n_slices<image.shape[0]:
                cl.enqueue_copy(cl_queue, image_in, image[i+n_slices:i+2*n_slices,:,:]).wait()
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

    def _run_threaded(self, float[:,:,:] image, int magnification, float ringRadius, int border, int radialityPositivityConstraint, int doIntensityWeighting):

        # CR
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int _magnification = magnification
        cdef int _border = border        
        
        cdef int rowsM = <int>(rows * _magnification)
        cdef int colsM = <int>(cols * _magnification)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col
        cdef float shift_row = 0.0
        cdef float shift_col = 0.0

        with nogil:
            for f in range(nFrames):
                for j in prange(colsM):
                    col = j / _magnification - shift_col
                    for i in range(rowsM):
                        row = i / _magnification - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        # RAD
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
        
        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], rows, cols)
                 for j in prange((1 + _border) * _magnification, (rows - 1 - _border) * _magnification):
                    for i in range((1 + _border) * _magnification, (cols - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols) * _image_out[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols)

        return np.asarray(imRad)



    def _run_threaded_dynamic(self, float[:,:,:] image, int magnification, float ringRadius, int border, int radialityPositivityConstraint, int doIntensityWeighting):
        # CR
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int _magnification = magnification
        cdef int _border = border        
        
        cdef int rowsM = <int>(rows * _magnification)
        cdef int colsM = <int>(cols * _magnification)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col
        cdef float shift_row = 0.0
        cdef float shift_col = 0.0

        with nogil:
            for f in range(nFrames):
                for j in prange(colsM,schedule="dynamic"):
                    col = j / _magnification - shift_col
                    for i in range(rowsM):
                        row = i / _magnification - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        # RAD
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

        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], rows, cols)
                 for j in prange((1 + _border) * _magnification, (rows - 1 - _border) * _magnification,schedule="dynamic"):
                    for i in range((1 + _border) * _magnification, (cols - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols) * _image_out[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols)

        return np.asarray(imRad)

    def _run_threaded_static(self, float[:,:,:] image, int magnification, float ringRadius, int border, int radialityPositivityConstraint, int doIntensityWeighting):
        # CR
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int _magnification = magnification
        cdef int _border = border        
        
        cdef int rowsM = <int>(rows * _magnification)
        cdef int colsM = <int>(cols * _magnification)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col
        cdef float shift_row = 0.0
        cdef float shift_col = 0.0

        with nogil:
            for f in range(nFrames):
                for j in prange(colsM,schedule="static"):
                    col = j / _magnification - shift_col
                    for i in range(rowsM):
                        row = i / _magnification - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        # RAD
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
        
        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], rows, cols)
                 for j in prange((1 + _border) * _magnification, (rows - 1 - _border) * _magnification,schedule="static"):
                    for i in range((1 + _border) * _magnification, (cols - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols) * _image_out[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols)

        return np.asarray(imRad)

    def _run_threaded_guided(self, float[:,:,:] image, int magnification, float ringRadius, int border, int radialityPositivityConstraint, int doIntensityWeighting):
        # CR
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int _magnification = magnification
        cdef int _border = border        
        
        cdef int rowsM = <int>(rows * _magnification)
        cdef int colsM = <int>(cols * _magnification)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col
        cdef float shift_row = 0.0
        cdef float shift_col = 0.0

        with nogil:
            for f in range(nFrames):
                for j in prange(colsM,schedule="guided"):
                    col = j / _magnification - shift_col
                    for i in range(rowsM):
                        row = i / _magnification - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        # RAD
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
        
        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], rows, cols)
                 for j in prange((1 + _border) * _magnification, (rows - 1 - _border) * _magnification,schedule="guided"):
                    for i in range((1 + _border) * _magnification, (cols - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols) * _image_out[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols)

        return np.asarray(imRad)

    def _run_unthreaded(self, float[:,:,:] image, int magnification, float ringRadius, int border, int radialityPositivityConstraint, int doIntensityWeighting):

        # CR
        cdef int nFrames = image.shape[0]
        cdef int rows = image.shape[1]
        cdef int cols = image.shape[2]
        cdef int _magnification = magnification
        cdef int _border = border        
        
        cdef int rowsM = <int>(rows * _magnification)
        cdef int colsM = <int>(cols * _magnification)

        image_out = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)
        cdef float[:,:,:] _image_out = image_out
        cdef float[:,:,:] _image_in = image

        cdef int f, i, j
        cdef float row, col
        cdef float shift_row = 0.0
        cdef float shift_col = 0.0

        with nogil:
            for f in range(nFrames):
                for j in range(colsM):
                    col = j / _magnification - shift_col
                    for i in range(rowsM):
                        row = i / _magnification - shift_row
                        _image_out[f, i, j] = _c_interpolate(&_image_in[f, 0, 0], row, col, rows, cols)

        # RAD
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
        
        cdef float [:,:,:] imGx = np.zeros_like(image) 
        cdef float [:,:,:] imGy = np.zeros_like(image)
        cdef float [:,:,:] imRad = np.zeros((nFrames, rowsM, colsM), dtype=np.float32)

        with nogil:
            for f in range(nFrames):
                 _c_gradient_radiality(&image[f,0,0], &imGx[f,0,0], &imGy[f,0,0], rows, cols)
                 for j in range((1 + _border) * _magnification, (rows - 1 - _border) * _magnification):
                    for i in range((1 + _border) * _magnification, (cols - 1 - _border) * _magnification):
                        if _doIntensityWeighting:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols) * _image_out[f, j, i]
                        else:
                            imRad[f,j,i] = _c_calculate_radiality_per_subpixel(i, j, &imGx[f,0,0], &imGy[f,0,0], xRingCoordinates, yRingCoordinates, _magnification, _ringRadius, nRingCoordinates, _radialityPositivityConstraint, rows, cols)

        return np.asarray(imRad)
