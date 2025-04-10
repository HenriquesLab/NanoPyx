# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False
import numpy as np
import math

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
    float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float PSF_voxel_ratio, float Gx_Gy_MAGNIFICATION, float Gz_MAGNIFICATION, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) nogil

class eSRRF3D(LiquidEngine):
    """
    eSRRF 3D using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "eSRRF_3D"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)
        self._gradients_s_interpolated = None
        self._gradients_r_interpolated = None
        self._gradients_c_interpolated = None
        self.keep_gradients = False
        self.keep_interpolated = False
        self._img_interpolated = None

    def run(self, image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True, keep_gradients=False, keep_interpolated = False, run_type=None):
        self.keep_gradients = keep_gradients
        self.keep_interpolated = keep_interpolated
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

    def _run_threaded(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True):
        """
        @cpu
        @threaded
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius * PSF_voxel_ratio / 2.355 # Taking voxel size into account
        cdef float fwhm_z = radius * PSF_voxel_ratio
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int Gx_Gy_MAGNIFICATION = 2
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

            if self.keep_gradients:
                self._gradients_s_interpolated = gradients_s_interpolated.copy()
                self._gradients_r_interpolated = gradients_r_interpolated.copy()
                self._gradients_c_interpolated = gradients_c_interpolated.copy()

            if self.keep_interpolated:
                self._img_interpolated = img_dum.copy()

            with nogil:
                for sM in range(0, n_slices_mag):
                    for rM in prange(0, n_rows_mag):
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val * image_interpolated[sM, rM, cM]
                            else:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val
        
        return np.asarray(rgc_map)
    def _run_threaded_guided(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True):
        """
        @cpu
        @threaded
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius * PSF_voxel_ratio / 2.355 # Taking voxel size into account
        cdef float fwhm_z = radius * PSF_voxel_ratio
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int Gx_Gy_MAGNIFICATION = 2
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

            if self.keep_gradients:
                self._gradients_s_interpolated = gradients_s_interpolated.copy()
                self._gradients_r_interpolated = gradients_r_interpolated.copy()
                self._gradients_c_interpolated = gradients_c_interpolated.copy()

            if self.keep_interpolated:
                self._img_interpolated = img_dum.copy()

            with nogil:
                for sM in range(0, n_slices_mag):
                    for rM in prange(0, n_rows_mag, schedule="guided"):
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val * image_interpolated[sM, rM, cM]
                            else:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val
        
        return np.asarray(rgc_map)
    def _run_threaded_dynamic(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True):
        """
        @cpu
        @threaded
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius * PSF_voxel_ratio / 2.355 # Taking voxel size into account
        cdef float fwhm_z = radius * PSF_voxel_ratio
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int Gx_Gy_MAGNIFICATION = 2
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

            if self.keep_gradients:
                self._gradients_s_interpolated = gradients_s_interpolated.copy()
                self._gradients_r_interpolated = gradients_r_interpolated.copy()
                self._gradients_c_interpolated = gradients_c_interpolated.copy()

            if self.keep_interpolated:
                self._img_interpolated = img_dum.copy()

            with nogil:
                for sM in range(0, n_slices_mag):
                    for rM in prange(0, n_rows_mag, schedule="dynamic"):
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val * image_interpolated[sM, rM, cM]
                            else:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val
        
        return np.asarray(rgc_map)
    def _run_threaded_static(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True):
        """
        @cpu
        @threaded
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius * PSF_voxel_ratio / 2.355 # Taking voxel size into account
        cdef float fwhm_z = radius * PSF_voxel_ratio
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int Gx_Gy_MAGNIFICATION = 2
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

            if self.keep_gradients:
                self._gradients_s_interpolated = gradients_s_interpolated.copy()
                self._gradients_r_interpolated = gradients_r_interpolated.copy()
                self._gradients_c_interpolated = gradients_c_interpolated.copy()

            if self.keep_interpolated:
                self._img_interpolated = img_dum.copy()

            with nogil:
                for sM in range(0, n_slices_mag):
                    for rM in prange(0, n_rows_mag, schedule="static"):
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val * image_interpolated[sM, rM, cM]
                            else:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val
        
        return np.asarray(rgc_map)
    def _run_unthreaded(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, PSF_voxel_ratio: float = 4.0, sensitivity: float = 1, doIntensityWeighting: bool = True):
        """
        @cpu
        @cython
        """
        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius * PSF_voxel_ratio / 2.355 # Taking voxel size into account
        cdef float fwhm_z = radius * PSF_voxel_ratio
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int Gx_Gy_MAGNIFICATION = 2
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

            if self.keep_gradients:
                self._gradients_s_interpolated = gradients_s_interpolated.copy()
                self._gradients_r_interpolated = gradients_r_interpolated.copy()
                self._gradients_c_interpolated = gradients_c_interpolated.copy()

            if self.keep_interpolated:
                self._img_interpolated = img_dum.copy()

            with nogil:
                for sM in range(0, n_slices_mag):
                    for rM in range(0, n_rows_mag):
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val * image_interpolated[sM, rM, cM]
                            else:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, PSF_voxel_ratio, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                rgc_map[f, sM, rM, cM] = rgc_val
        
        return np.asarray(rgc_map)

    def get_gradients(self):
        if self._gradients_c_interpolated is None or self._gradients_r_interpolated is None or self._gradients_s_interpolated is None:
            print("Gradients not yet calculated")
        else:
            return self._gradients_c_interpolated, self._gradients_r_interpolated, self._gradients_s_interpolated

    def get_interpolated_image(self):
        return self._img_interpolated