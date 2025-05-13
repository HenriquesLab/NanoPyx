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

from scipy.stats import pearsonr as pearson_correlation

cdef extern from "_c_gradients.h":
    void _c_gradient_3d(float* image, float* imGc, float* imGr, float* imGs, int slices, int rows, int cols) nogil

cdef extern from "_c_sr_radial_gradient_convergence.h":
    float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float voxel_ratio, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) nogil

class eSRRF3D(LiquidEngine):
    """
    eSRRF 3D using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "eSRRF_3D"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image, magnification_xy: int = 2, magnification_z: int = 2, radius: float = 1.5, radius_z: float = 0.5, voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True, run_type=None):
        # TODO: complete and check _run inputs, need to complete variables?
        if len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        if len(image.shape) != 4:
            print("Warning:image must either be 3D or 4D. If 3D, it will be reshaped to 4D.")
            return None
        if radius * 2 > (image.shape[2]) / 2 or radius * 2 > (image.shape[3] / 2):
            print("Warning: Radius is too big for the image. Half the radius must be smaller than both half the number of columns and half number of rows of the image.")
            return None
        if radius_z * 2 > image.shape[1] / 2:
            print("Warning: Radius_z is too big for the image. Half the radius_z must be smaller than half of number of Z planes.")
            return None
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, radius_z=radius_z, voxel_ratio=voxel_ratio, sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting, run_type=run_type)

    def benchmark(self, image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, radius_z: float = 0.5, voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return super().benchmark(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, radius_z=radius_z, voxel_ratio=voxel_ratio,sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return super().benchmark(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, radius_z=radius_z, voxel_ratio=voxel_ratio, sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting)

    % for sch in schedulers:
    def _run_${sch}(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, radius_z: float = 0.5, voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True):
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
        cdef int margin = int(2 * radius) * magnification_xy
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius_z * voxel_ratio / 2.355 # Taking voxel size into account
        cdef int margin_z = int(2 * radius_z) * magnification_z
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef float fwhm = radius
        cdef float fwhm_z = radius_z
        cdef int _magnification_xy = magnification_xy
        cdef int _magnification_z = magnification_z
        cdef float _voxel_ratio = voxel_ratio
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int n_frames, n_slices, n_rows, n_cols, n_slices_mag, n_rows_mag, n_cols_mag
        n_frames, n_slices, n_rows, n_cols = image.shape[0], image.shape[1], image.shape[2], image.shape[3]
        n_slices_mag = n_slices * _magnification_z
        n_rows_mag = n_rows * _magnification_xy
        n_cols_mag = n_cols * _magnification_xy
        
        cdef float[:, :, :] rgc_mean
        # create all necessary arrays
        cdef float[:, :, :] rgc_out = np.zeros((n_slices_mag, n_rows_mag, n_cols_mag), dtype=np.float32)
        if mode == "std":
            rgc_mean = np.zeros((n_slices_mag, n_rows_mag, n_cols_mag), dtype=np.float32)
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
                for sM in range(margin_z, n_slices_mag-margin_z):
                    % if sch=="unthreaded":
                    for rM in range(margin, n_rows_mag-margin):
                    % elif sch=="threaded":
                    for rM in prange(margin, n_rows_mag-margin):
                    % else:
                    for rM in prange(margin, n_rows_mag-margin, schedule="${sch.split('_')[1]}"):
                    % endif
                        for cM in range(margin, n_cols_mag-margin):
                            rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_col_mag[0,0,0], &gradients_row_mag[0,0,0], &gradients_slices_mag[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, _voxel_ratio, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                            if _doIntensityWeighting:
                                rgc_val = rgc_val * image_interpolated[sM, rM, cM]
                            if mode == "average":
                                rgc_out[sM, rM, cM] = rgc_out[sM, rM, cM] + (rgc_val - rgc_out[sM, rM, cM]) / (f_i + 1)
                            elif mode == "std":
                                delta = rgc_val - rgc_mean[sM, rM, cM]
                                rgc_mean[sM, rM, cM] += delta / (f_i + 1)
                                delta_2 = rgc_val - rgc_mean[sM, rM, cM]
                                rgc_out[sM, rM, cM] += delta * delta_2
        if mode == "std":
            rgc_out = np.sqrt(np.asarray(rgc_out) / n_frames)
            return rgc_out
        else:
            return np.asarray(rgc_out)
    % endfor

    def _run_opencl(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, radius_z: float = 0.5, voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True, device=None, mem_div=1):
        """
        @gpu
        """
        # select cl device
        if device is None:
            device = _fastest_device
        
        # create cl context and queue
        cl_ctx = cl.Context([device['device']])
        dc = device['device']
        cl_queue = cl.CommandQueue(cl_ctx)

        output_shape = (image.shape[1] * magnification_z, image.shape[2] * magnification_xy, image.shape[3] * magnification_xy)
        output_array = np.zeros(output_shape, dtype=np.float32)

        # create input cl buffers
        input_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(image))
        intermediate_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=image.shape[1] * image.shape[2] * magnification_xy * image.shape[3] * magnification_xy * np.dtype(np.float32).itemsize)
        input_magnified_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=output_array.nbytes)

        # create gradient buffers
        slices_gradient_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=image.shape[1] * image.shape[2] * image.shape[3] * np.dtype(np.float32).itemsize)
        rows_gradient_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=image.shape[1] * image.shape[2] * image.shape[3] * np.dtype(np.float32).itemsize)
        cols_gradient_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=image.shape[1] * image.shape[2] * image.shape[3] * np.dtype(np.float32).itemsize)
        slices_gradient_magnified_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=output_array.nbytes)
        rows_gradient_magnified_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=output_array.nbytes)
        cols_gradient_magnified_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=output_array.nbytes)

        # create rgc cl buffer
        rgc_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=output_array.nbytes)

        # create output cl buffer
        output_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=output_array.nbytes)
        if mode == "std":
            mean_buffer = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=output_array.nbytes)

        # create cl code, program and kernels
        cl_code = self._get_cl_code("_le_esrrf3d_.cl", device["DP"])
        cl_prg = cl.Program(cl_ctx, cl_code).build(options=["-cl-fast-relaxed-math", "-cl-mad-enable"])
        interpolate_xy_kernel = cl_prg.interpolate_xy_2d
        interpolate_z_kernel = cl_prg.interpolate_z_1d
        gradients_kernel = cl_prg.gradients_3d
        rgc_kernel = cl_prg.calculate_rgc3D
        if mode == "average":
            time_projection_kernel = cl_prg.time_projection_average
        elif mode == "std":
            time_projection_kernel = cl_prg.time_projection_std
        else:
            raise ValueError("Invalid mode. Use 'average' or 'std'.")

        # set margins
        margin = int(2 * radius) * magnification_xy
        margin_z = int(2 * radius_z) * magnification_z
        lowest_slice = margin_z
        highest_slice = output_shape[0] - margin_z
        lowest_row = margin
        highest_row = output_shape[1] - margin
        lowest_col = margin
        highest_col = output_shape[2] - margin

        # set constants
        cdef float sigma = radius / 2.355
        cdef float tss = 2 * sigma * sigma
        cdef float tso = 2 * sigma + 1
        cdef float sigma_z = radius_z * voxel_ratio / 2.355 # Taking voxel size into account
        cdef float tss_z = 2 * sigma_z * sigma_z
        cdef float tso_z = 2 * sigma_z + 1

        cdef float frame_div = 0.0

        # loop over frames:
        for frame_i in range(image.shape[0]):
            # interpolate image
            interpolate_xy_kernel(
                cl_queue,
                (image.shape[1], output_shape[1], output_shape[2]),
                None,
                input_buffer,
                intermediate_buffer,
                np.float32(magnification_xy),
                np.int32(frame_i),
            ).wait()

            interpolate_z_kernel(
                cl_queue,
                (output_shape[0], output_shape[1], output_shape[2]),
                None,
                intermediate_buffer,
                input_magnified_buffer,
                np.float32(magnification_z),
                np.int32(0),
            ).wait()

            # calculate gradients
            gradients_kernel(
                cl_queue,
                (image.shape[1], image.shape[2], image.shape[3]),
                None,
                input_buffer,
                slices_gradient_buffer,
                cols_gradient_buffer,
                rows_gradient_buffer,
                np.int32(image.shape[1]),
                np.int32(image.shape[2]),
                np.int32(image.shape[3]),
                np.int32(frame_i),
            ).wait()

            # interpolate gradients
            interpolate_xy_kernel(
                cl_queue,
                (image.shape[1], output_shape[1], output_shape[2]),
                None,
                slices_gradient_buffer,
                intermediate_buffer,
                np.float32(magnification_xy),
                np.int32(0),
            ).wait()
            interpolate_z_kernel(
                cl_queue,
                (output_shape[0], output_shape[1], output_shape[2]),
                None,
                intermediate_buffer,
                slices_gradient_magnified_buffer,
                np.float32(magnification_z),
                np.int32(0),
            ).wait()

            interpolate_xy_kernel(
                cl_queue,
                (image.shape[1], output_shape[1], output_shape[2]),
                None,
                rows_gradient_buffer,
                intermediate_buffer,
                np.float32(magnification_xy),
                np.int32(0),
            ).wait()
            interpolate_z_kernel(
                cl_queue,
                (output_shape[0], output_shape[1], output_shape[2]),
                None,
                intermediate_buffer,
                rows_gradient_magnified_buffer,
                np.float32(magnification_z),
                np.int32(0),
            ).wait()

            interpolate_xy_kernel(
                cl_queue,
                (image.shape[1], output_shape[1], output_shape[2]),
                None,
                cols_gradient_buffer,
                intermediate_buffer,
                np.float32(magnification_xy),
                np.int32(0),
            ).wait()
            interpolate_z_kernel(
                cl_queue,
                (output_shape[0], output_shape[1], output_shape[2]),
                None,
                intermediate_buffer,
                cols_gradient_magnified_buffer,
                np.float32(magnification_z),
                np.int32(0),
            ).wait()

            # calculate rgc

            rgc_kernel(
                cl_queue,
                (highest_slice - lowest_slice, highest_row - lowest_row, highest_col - lowest_col),
                None,
                slices_gradient_magnified_buffer,
                rows_gradient_magnified_buffer,
                cols_gradient_magnified_buffer,
                input_magnified_buffer,
                rgc_buffer,
                np.int32(output_shape[0]),
                np.int32(output_shape[1]),
                np.int32(output_shape[2]),
                np.int32(magnification_xy),
                np.int32(magnification_z),
                np.float32(voxel_ratio),
                np.float32(radius),
                np.float32(radius_z),
                np.float32(tso),
                np.float32(tss),
                np.float32(tso_z),
                np.float32(tss_z),
                np.float32(sensitivity),
                np.int32(doIntensityWeighting),
                np.int32(frame_i),
            ).wait()
            
            if mode == "std":
                time_projection_kernel(
                    cl_queue,
                    (output_shape[0], output_shape[1], output_shape[2]),
                    None,
                    rgc_buffer,
                    mean_buffer,
                    output_buffer,
                    np.int32(frame_i),
                ).wait()
            else:
                time_projection_kernel(
                    cl_queue,
                    (output_shape[0], output_shape[1], output_shape[2]),
                    None,
                    rgc_buffer,
                    output_buffer,
                    np.int32(frame_i),
                ).wait()
            cl_queue.finish()
        cl.enqueue_copy(cl_queue, output_array, output_buffer).wait()

        if mode == "std":
            return np.asarray(np.sqrt(output_array / image.shape[0]))
        else:
            return np.asarray(output_array)


    def _compare_runs(self, output_1, output_2):
        """@public"""
        if output_1.ndim > 2:
            pcc = 0
            count = 0 
            for i in range(output_1.shape[0]):
                pccresult = pearson_correlation(output_1[i, :, :].flatten(), output_2[i, :, :].flatten()).statistic

                if np.isnan(pccresult):
                    continue
                else:
                    count += 1
                    # calculate pcc for each frame
                    pcc += (pccresult-pcc) / count
        else:
            pcc = pearson_correlation(output_1.flatten(), output_2.flatten()).statistic

        if pcc > 0.8:
            return True
        else:
            return False