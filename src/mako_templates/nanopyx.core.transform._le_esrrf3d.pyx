<%!
schedulers = ['threaded','threaded_guided','threaded_dynamic','threaded_static', 'unthreaded']
%># cython: infer_types=True, wraparound=True, nonecheck=True, boundscheck=True, cdivision=True, language_level=3, profile=True, autogen_pxd=False
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

cdef extern from "_c_gradients.h":
    void _c_gradient_3d(float* image, float* imGc, float* imGr, float* imGs, int slices, int rows, int cols) nogil

cdef extern from "_c_sr_radial_gradient_convergence.h":
    float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float PSF_voxel_ratio, float fwhm, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) nogil

class eSRRF3D(LiquidEngine):
    """
    eSRRF 3D using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "eSRRF_3D"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type=None):
        # TODO: complete and check _run inputs, need to complete variables?
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, PSF_voxel_ratio=PSF_voxel_ratio, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, PSF_voxel_ratio=PSF_voxel_ratio, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)

    def benchmark(self, image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, PSF_voxel_ratio=PSF_voxel_ratio,sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return super().benchmark(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, PSF_voxel_ratio=PSF_voxel_ratio, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting)

    % for sch in schedulers:
    def _run_${sch}(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True):
        """
        @cpu
        % if sch!='unthreaded':
        @threaded
        % endif
        @cython
        """
        time_start = time.time()
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius * PSF_voxel_ratio / 2.355 # Taking voxel size into account
        cdef float fwhm_z = radius * PSF_voxel_ratio
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int Gx_Gy_MAGNIFICATION = 1
        cdef int _magnification_xy = magnification_xy
        cdef int _magnification_z = magnification_z
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int n_frames, n_slices, n_rows, n_cols, n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum
        n_frames, n_slices, n_rows, n_cols = image.shape[0], image.shape[1], image.shape[2], image.shape[3]

        cdef float[:, :, :, :] rgc_map = np.zeros((n_frames, n_slices*magnification_z, n_rows*magnification_xy, n_cols*magnification_xy), dtype=np.float32)
        
        cdef float[:, :, :] image_interpolated, gradients_s, gradients_r, gradients_c, gradients_s_interpolated, gradients_r_interpolated, gradients_c_interpolated, padded, img_dum

        cdef int f, n_slices_mag, n_rows_mag, n_cols_mag, sM, rM, cM, z0

        cdef float rgc_val, zcof

        for f in range(n_frames):
            image_interpolated = interpolate_3d_zlinear(image[f,:,:,:], _magnification_xy, _magnification_z)

            n_slices_mag, n_rows_mag, n_cols_mag = image_interpolated.shape[0], image_interpolated.shape[1], image_interpolated.shape[2]

            img_dum = interpolate_3d_zlinear(image[f], Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION) 
            n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum = img_dum.shape[0], img_dum.shape[1], img_dum.shape[2]

            gradients_c = np.zeros((n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum), dtype=np.float32)
            gradients_r = np.zeros((n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum), dtype=np.float32)
            gradients_s = np.zeros((n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum), dtype=np.float32)
            with nogil:
                _c_gradient_3d(&img_dum[0, 0, 0], &gradients_c[0, 0, 0], &gradients_r[0, 0, 0], &gradients_s[0, 0, 0], n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum)

            gradients_s_interpolated = interpolate_3d_zlinear(gradients_s, _magnification_xy, _magnification_z)
            gradients_r_interpolated = interpolate_3d_zlinear(gradients_r, _magnification_xy, _magnification_z)
            gradients_c_interpolated = interpolate_3d_zlinear(gradients_c, _magnification_xy, _magnification_z)

            with nogil:
                for sM in range(0, n_slices_mag):
                    % if sch=="unthreaded":
                    for rM in range(0, n_rows_mag):
                    % elif sch=="threaded":
                    for rM in prange(0, n_rows_mag):
                    % else:
                    for rM in prange(0, n_rows_mag, schedule="${sch.split('_')[1]}"):
                    % endif
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, fwhm, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val * image_interpolated[sM, rM, cM]
                            else:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, fwhm, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val
        
        return np.asarray(rgc_map)
    % endfor


class eSRRF3D_v2(LiquidEngine):
    """
    eSRRF 3D using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "eSRRF_3D_v2"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image, magnification_xy: int = 2, magnification_z: int = 2, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True, run_type=None):
        # TODO: complete and check _run inputs, need to complete variables?
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, PSF_voxel_ratio=PSF_voxel_ratio, sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting, run_type=run_type)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, PSF_voxel_ratio=PSF_voxel_ratio, sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting, run_type=run_type)

    def benchmark(self, image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return super().benchmark(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, PSF_voxel_ratio=PSF_voxel_ratio,sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return super().benchmark(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, PSF_voxel_ratio=PSF_voxel_ratio, sensitivity=sensitivity, mode=mode, doIntensityWeighting=doIntensityWeighting)

    % for sch in schedulers:
    def _run_${sch}(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, mode: str = "average", doIntensityWeighting: bool = True):
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
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius * PSF_voxel_ratio / 2.355 # Taking voxel size into account
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int _magnification_xy = magnification_xy
        cdef int _magnification_z = magnification_z
        cdef float _PSF_voxel_ratio = PSF_voxel_ratio
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
            image_interpolated = interpolate_3d_zlinear(image[f_i,:,:,:], magnification_xy, _magnification_z)

            # calculate gradients
            _c_gradient_3d(&image_interpolated[0, 0, 0], &gradients_col_mag[0, 0, 0], &gradients_row_mag[0, 0, 0], &gradients_slices_mag[0, 0, 0], n_slices_mag, n_rows_mag, n_cols_mag)

            # interpolate gradients
            # gradients_slices_mag = interpolate_3d_zlinear(gradients_slices, _magnification_xy, _magnification_z)
            # gradients_row_mag = interpolate_3d_zlinear(gradients_row, _magnification_xy, _magnification_z)
            # gradients_col_mag = interpolate_3d_zlinear(gradients_col, _magnification_xy, _magnification_z)

            with nogil:
                for sM in range(0, n_slices_mag):
                    % if sch=="unthreaded":
                    for rM in range(0, n_rows_mag):
                    % elif sch=="threaded":
                    for rM in prange(0, n_rows_mag):
                    % else:
                    for rM in prange(0, n_rows_mag, schedule="${sch.split('_')[1]}"):
                    % endif
                        for cM in range(0, n_cols_mag):
                            rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_col_mag[0,0,0], &gradients_row_mag[0,0,0], &gradients_slices_mag[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, _PSF_voxel_ratio, radius, tSO, tSO_z, tSS, tSS_z, sensitivity)
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