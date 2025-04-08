<%!
schedulers = ['threaded','threaded_guided','threaded_dynamic','threaded_static', 'unthreaded']
%># cython: infer_types=True, wraparound=True, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False
import numpy as np
import math
import time

cimport numpy as np
from libc.math cimport floor

from cython.parallel import prange

from .sr_temporal_correlations import calculate_eSRRF_temporal_correlations

from ._interpolation import interpolate_3d, interpolate_3d_zlinear
from ._le_interpolation_catmull_rom import ShiftAndMagnify
from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array, _fastest_device

cdef extern from "_c_gradients.h":
    void _c_gradient_3d(float* image, float* imGc, float* imGr, float* imGs, int slices, int rows, int cols) nogil

cdef extern from "_c_sr_radial_gradient_convergence.h":
    float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float voxel_ratio, float fwhm, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) nogil

class eSRRF3D(LiquidEngine):
    """
    eSRRF 3D using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "eSRRF_3D"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image, magnification_xy: int = 2, magnification_z: int = 2, radius: float = 1.5, voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True, run_type=None):
        # TODO: complete and check _run inputs, need to complete variables?
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, voxel_ratio=voxel_ratio, sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting, run_type=run_type)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, voxel_ratio=voxel_ratio, sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting, run_type=run_type)

    def benchmark(self, image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return super().benchmark(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, voxel_ratio=voxel_ratio,sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return super().benchmark(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, voxel_ratio=voxel_ratio, sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting)

    % for sch in schedulers:
    def _run_${sch}(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True):
        """
        @cpu
        % if sch!='unthreaded':
        @threaded
        % endif
        @cython
        """

        time_start = time.time()
        # calculate all constants
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef int margin = int(2 * radius)
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius * voxel_ratio / 2.355 # Taking voxel size into account
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int _magnification_xy = magnification_xy
        cdef int _magnification_z = magnification_z
        cdef float _voxel_ratio = voxel_ratio
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int n_frames, n_slices, n_rows, n_cols, n_slices_mag, n_rows_mag, n_cols_mag
        n_frames, n_slices, n_rows, n_cols = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
        n_slices_mag = n_slices * _magnification_z
        n_rows_mag = n_rows * _magnification_xy
        n_cols_mag = n_cols * _magnification_xy
        
        # create all necessary arrays
        cdef float[:, :, :] rgc_avg = np.zeros((n_slices_mag, n_rows_mag, n_cols_mag), dtype=np.float32)
        cdef float[:, :, :] rgc_std
        if mode == "std":
            rgc_std = np.zeros((n_slices_mag, n_rows_mag, n_cols_mag), dtype=np.float32)
        cdef float[:, :, :] image_interpolated = np.zeros((n_slices_mag, n_rows_mag, n_cols_mag), dtype=np.float32)
        cdef float[:, :, :] gradients_col = np.zeros((n_slices, n_rows, n_cols), dtype=np.float32)
        cdef float[:, :, :] gradients_row = np.zeros((n_slices, n_rows, n_cols), dtype=np.float32)
        cdef float[:, :, :] gradients_slices = np.zeros((n_slices, n_rows, n_cols), dtype=np.float32)
        cdef float[:, :, :] gradients_col_mag = np.zeros((n_slices_mag, n_rows_mag, n_cols_mag), dtype=np.float32)
        cdef float[:, :, :] gradients_row_mag = np.zeros((n_slices_mag, n_rows_mag, n_cols_mag), dtype=np.float32)
        cdef float[:, :, :] gradients_slices_mag = np.zeros((n_slices_mag, n_rows_mag, n_cols_mag), dtype=np.float32)
        cdef float delta, delta_2, rgc_val
        cdef int f_i, sM, rM, cM

        for f_i in range(n_frames):
            # interpolate frame
            image_interpolated = interpolate_3d_zlinear(image[f_i,:,:,:], _magnification_xy, _magnification_z)

            with nogil:
                # calculate gradients
                _c_gradient_3d(&image[f_i, 0, 0, 0], &gradients_col[0, 0, 0], &gradients_row[0, 0, 0], &gradients_slices[0, 0, 0], n_slices, n_rows, n_cols)

            # interpolate gradients
            gradients_slices_mag = interpolate_3d_zlinear(gradients_slices, _magnification_xy, _magnification_z)
            gradients_row_mag = interpolate_3d_zlinear(gradients_row, _magnification_xy, _magnification_z)
            gradients_col_mag = interpolate_3d_zlinear(gradients_col, _magnification_xy, _magnification_z)

            with nogil:
                for sM in range(margin, n_slices_mag-margin):
                    % if sch=="unthreaded":
                    for rM in range(margin, n_rows_mag-margin):
                    % elif sch=="threaded":
                    for rM in prange(margin, n_rows_mag-margin):
                    % else:
                    for rM in prange(margin, n_rows_mag-margin, schedule="${sch.split('_')[1]}"):
                    % endif
                        for cM in range(margin, n_cols_mag-margin):
                            rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_col_mag[0,0,0], &gradients_row_mag[0,0,0], &gradients_slices_mag[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, _voxel_ratio, fwhm, tSO, tSO_z, tSS, tSS_z, sensitivity)
                            if _doIntensityWeighting:
                                rgc_val = rgc_val * image_interpolated[sM, rM, cM]
                            if mode == "average":
                                rgc_avg[sM, rM, cM] = rgc_avg[sM, rM, cM] + (rgc_val - rgc_avg[sM, rM, cM]) / (f_i + 1)
                            elif mode == "std":
                                delta = rgc_val - rgc_avg[sM, rM, cM] 
                                rgc_avg[sM, rM, cM] = rgc_avg[sM, rM, cM] + (delta) / (f_i + 1)
                                delta_2 = rgc_val - rgc_avg[sM, rM, cM]
                                rgc_std[sM, rM, cM] = rgc_std[sM, rM, cM] + (delta * delta_2)
        if mode == "std":
            rgc_std = np.sqrt(np.asarray(rgc_std) / n_frames)
            return rgc_std
        else:
            return np.asarray(rgc_avg)
    % endfor

    def _run_opencl(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True, device=None, mem_div=1):
        """
        @gpu
        """

        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        output_shape = (image.shape[1] * magnification_z, image.shape[2] * magnification_xy, image.shape[3] * magnification_xy)

        output_image = np.zeros(output_shape, dtype=np.float32)
        tmp_slice = np.zeros(output_shape, dtype=np.float32)

        #max_slices = int((dc.global_mem_size - (3 * (image[0, :, :, :].nbytes) + 4 * (output_image.nbytes)) // (image[0, :, :, :].nbytes)) // mem_div)
        #max_slices = self._check_max_slices(image, max_slices)

        max_slices = 1

        mf = cl.mem_flags

        # create cl buffer for input image
        input_cl = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(image[:, :, :, :].nbytes, dc, max_slices))
        cl.enqueue_copy(cl_queue, input_cl, image[:,:,:, :]).wait()

        # create magnified input image buffer
        magnified_cl = cl.Buffer(cl_ctx, mf.READ_ONLY, self._check_max_buffer_size(output_image.nbytes, dc, max_slices))

        # create gradients buffers
        col_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(image[0, :, :, :].nbytes, dc, max_slices))
        row_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(image[0, :, :, :].nbytes, dc, max_slices))
        z_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(image[0, :, :, :].nbytes, dc, max_slices))

        # create magnified gradients buffers
        col_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(output_image.nbytes, dc, max_slices))
        row_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(output_image.nbytes, dc, max_slices))
        z_magnified_gradients_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(output_image.nbytes, dc, max_slices))

        # create the output buffer
        output_cl = cl.Buffer(cl_ctx, mf.READ_WRITE, self._check_max_buffer_size(output_image.nbytes, dc, max_slices))

        esrrf3d_cl_code = self._get_cl_code("esrrf3d.cl", device["DP"])
        esrrf3d_cl_prg = cl.Program(cl_ctx, esrrf3d_cl_code).build(options=["-cl-fast-relaxed-math", "-cl-mad-enable"])
        interp_knl = esrrf3d_cl_prg.interpolate_3d
        grads_knl = esrrf3d_cl_prg.gradients_3d
        rgc_knl = esrrf3d_cl_prg.calculate_rgc3D

        margin = int(radius*2)
        lowest_row = margin # TODO discuss edges calculation
        highest_row = output_shape[1] - margin
        lowest_col = margin
        highest_col =  output_shape[2] - margin

        for f in range(image.shape[0]):

            interp_knl(
                cl_queue,
                (image.shape[1], image.shape[2], image.shape[3]),
                None,
                input_cl,
                magnified_cl,
                np.int32(magnification_xy),
                np.int32(magnification_z),
                np.int32(f)
            ).wait()


            grads_knl(
                cl_queue,
                (image.shape[1], image.shape[2], image.shape[3]),
                None,
                input_cl,
                col_gradients_cl,
                row_gradients_cl,
                z_gradients_cl,
                np.int32(f),
                np.int32(image.shape[1]),
                np.int32(image.shape[2]),
                np.int32(image.shape[3]),
            ).wait()

            interp_knl(
                cl_queue,
                (output_shape[0], output_shape[1], output_shape[2]),
                None,
                col_gradients_cl,
                col_magnified_gradients_cl,
                np.int32(magnification_xy),
                np.int32(magnification_z),
                np.int32(f)
            ).wait()

            interp_knl(
                cl_queue,
                (output_shape[0], output_shape[1], output_shape[2]),
                None,
                row_gradients_cl,
                row_magnified_gradients_cl,
                np.int32(magnification_xy),
                np.int32(magnification_z),
                np.int32(f)
            ).wait()

            interp_knl(
                cl_queue,
                (output_shape[0], output_shape[1], output_shape[2]),
                None,
                z_gradients_cl,
                z_magnified_gradients_cl,
                np.int32(magnification_xy),
                np.int32(magnification_z),
                np.int32(f)
            ).wait()

            rgc_knl(
                cl_queue,
                (output_shape[0], output_shape[1], output_shape[2]),
                None,
                col_magnified_gradients_cl,
                row_magnified_gradients_cl,
                z_magnified_gradients_cl,
                magnified_cl,
                tmp_slice,
                np.int32(output_shape[1]),
                np.int32(output_shape[2]),
                np.int32(output_shape[0]),
                np.int32(magnification_xy),
                np.int32(magnification_z),
                np.float32(voxel_ratio),
                np.float32(radius),
                np.float32(2 * (radius/2.355) + 1),
                np.float32(2 * (radius*voxel_ratio/2.355) + 1),
                np.float32(2 * (radius/2.355) * (radius/2.355)),
                np.float32(2 * (radius*voxel_ratio/2.355) * (radius*voxel_ratio/2.355)),
                np.int32(sensitivity),
                np.int32(doIntensityWeighting)

            ).wait()

            
            # TODO change tmp_slice to cl arrays
            if mode == "average":
                output_cl = output_cl + (tmp_slice - output_cl) / (f + 1)
            elif mode == "std":
                delta = tmp_slice - output_cl
                output_cl = output_cl + (delta) / (f + 1)
                delta_2 = tmp_slice - output_cl
                output_cl = output_cl + (delta * delta_2)


        cl.enqueue_copy(cl_queue, output_image, output_cl).wait()

        if mode == "std":
            output_image = np.sqrt(np.asarray(output_image) / image.shape[0])
            return output_image
        else:
            return np.asarray(output_image)