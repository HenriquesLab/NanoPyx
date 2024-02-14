import numpy as np
import math

cimport numpy as np
from libc.math cimport floor

from cython.parallel import prange

from .sr_temporal_correlations import calculate_eSRRF_temporal_correlations

from ._interpolation import interpolate_3d
from ._le_interpolation_catmull_rom import ShiftAndMagnify
from ...__liquid_engine__ import LiquidEngine

cdef extern from "_c_gradients.h":
    void _c_gradient_3d(float* image, float* imGc, float* imGr, float* imGs, int slices, int rows, int cols) nogil

cdef extern from "_c_sr_radial_gradient_convergence.h":
    float _c_calculate_rgc3D(int xM, int yM, int sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int magnification_xy, int magnification_z, float Gx_Gy_MAGNIFICATION, float Gz_MAGNIFICATION, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) nogil

class eSRRF3D(LiquidEngine):
    """
    eSRRF 3D using the NanoPyx Liquid Engine and running as a single task.
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "eSRRF_3D"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                        opencl_=False, unthreaded_=True, threaded_=False, threaded_static_=False, 
                        threaded_dynamic_=False, threaded_guided_=False)
        self._default_benchmarks = {'OpenCL': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 0.6182407079995755, 0.6122367919997487, 0.6093217500001629], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 0.3028541249987029, 0.2802749999991647, 0.2996904169995105]}, 'Threaded': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 11.06756008299999, 11.480632083001183, 11.368770667000717], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 2.490110000000641, 2.6822768330002873, 2.5428189579997706]}, 'Threaded_dynamic': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 9.366981125000166, 9.478710332999981, 10.355995709000126], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 2.020473291999224, 2.0464198749996285, 2.1126812500006054]}, 'Threaded_guided': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 9.59399004199986, 9.394610874998762, 10.42694429200128], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 2.1240765410002496, 2.1175940839984833, 2.127043415999651]}, 'Threaded_static': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 11.325379959000202, 11.439641291000953, 12.059574375000011], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 2.4486127919990395, 2.5282963329991617, 2.7061019579996355]}, 'Unthreaded': {"(['shape(100, 150, 150)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [16875000.0, 50.56610608399933, 50.84249891699983, 51.47774041699995], "(['shape(50, 100, 100)'], {'magnification': 5, 'radius': 1.5, 'sensitivity': 1.0, 'doIntensityWeighting': True})": [3750000.0, 10.916643833001217, 11.070652250000421, 11.099240584000654]}}
        self._gradients_s_interpolated = None
        self._gradients_r_interpolated = None
        self._gradients_c_interpolated = None
        self.keep_gradients = False

    def run(self, image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, radius_z: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type=None, keep_gradients=False):
        self.keep_gradients = keep_gradients
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)
    
    def _run_unthreaded(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, radius_z: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type="Unthreaded"):

        interpolator = ShiftAndMagnify()

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius_z / 2.355
        cdef float fwhm_z = radius_z
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int Gx_Gy_MAGNIFICATION = 2
        cdef int _magnification_xy = magnification_xy
        cdef int _magnification_z = magnification_z
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int n_frames, n_slices, n_rows, n_cols, n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum
        n_frames, n_slices, n_rows, n_cols = image.shape[0], image.shape[1], image.shape[2], image.shape[3]

        cdef float[:, :, :, :] rgc_map = np.zeros((n_frames, n_slices*magnification_z, n_rows*magnification_xy, n_cols*magnification_xy), dtype=np.float32)

        cdef float[:, :, :, :] avg_rgc_map = np.zeros((n_frames, n_slices*magnification_z, n_rows*magnification_xy, n_cols*magnification_xy), dtype=np.float32)
        
        cdef float[:, :, :] image_interpolated, gradients_s, gradients_r, gradients_c, gradients_s_interpolated, gradients_r_interpolated, gradients_c_interpolated, padded, img_dum

        cdef int f, n_slices_mag, n_rows_mag, n_cols_mag, sM, rM, cM, z0
        cdef int w, n_windows, start_frame, end_frame

        cdef float rgc_val, zcof

        for f in range(n_frames):

            # image_interpolated = interpolator.run(image[f], 0, 0, _magnification_xy, _magnification_xy)
            image_interpolated = interpolate_3d(image[f], _magnification_xy, _magnification_z)
            n_slices_mag, n_rows_mag, n_cols_mag = image_interpolated.shape[0], image_interpolated.shape[1], image_interpolated.shape[2]

            gradients_c = np.zeros((n_slices, n_rows*Gx_Gy_MAGNIFICATION, n_cols*Gx_Gy_MAGNIFICATION), dtype=np.float32)
            gradients_r = np.zeros((n_slices, n_rows*Gx_Gy_MAGNIFICATION, n_cols*Gx_Gy_MAGNIFICATION), dtype=np.float32)
            gradients_s = np.zeros((n_slices, n_rows*Gx_Gy_MAGNIFICATION, n_cols*Gx_Gy_MAGNIFICATION), dtype=np.float32)

            img_dum = interpolator.run(image[f], 0, 0, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION)
            # img_dum = interpolate_3d(image[f], Gx_Gy_MAGNIFICATION, _magnification_z)
            n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum = img_dum.shape[0], img_dum.shape[1], img_dum.shape[2]

            with nogil:
                _c_gradient_3d(&img_dum[0, 0, 0], &gradients_c[0, 0, 0], &gradients_r[0, 0, 0], &gradients_s[0, 0, 0], n_slices, n_rows_mag_dum, n_cols_mag_dum)

            # gradients_s_interpolated = interpolator.run(gradients_s, 0, 0, _magnification_xy, _magnification_xy)
            # gradients_r_interpolated = interpolator.run(gradients_r, 0, 0, _magnification_xy, _magnification_xy)
            # gradients_c_interpolated = interpolator.run(gradients_c, 0, 0, _magnification_xy, _magnification_xy)

            gradients_s_interpolated = interpolate_3d(gradients_s, _magnification_xy, _magnification_z)
            gradients_r_interpolated = interpolate_3d(gradients_r, _magnification_xy, _magnification_z)
            gradients_c_interpolated = interpolate_3d(gradients_c, _magnification_xy, _magnification_z)

            if self.keep_gradients:
                self._gradients_s_interpolated = gradients_s_interpolated.copy()
                self._gradients_r_interpolated = gradients_r_interpolated.copy()
                self._gradients_c_interpolated = gradients_c_interpolated.copy()

            with nogil:
                for sM in range(0, n_slices_mag):
                    for rM in range(0, n_rows_mag):
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, Gx_Gy_MAGNIFICATION, 1, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                # zcof = (sM) / _magnification_z
                                # if (zcof < 0):
                                #     z0 = 0
                                # elif (zcof >= n_slices-1):
                                #     z0 = n_slices-1
                                # else:
                                #     z0 = <int> floor(zcof)
                                # z0 = z0 * _magnification_z
                                rgc_map[f, sM, rM, cM] = rgc_val * image_interpolated[sM, rM, cM]
                            else:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, Gx_Gy_MAGNIFICATION, 1, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                # zcof = (sM) / _magnification_z
                                # if (zcof < 0):
                                #     z0 = 0
                                # elif (zcof >= n_slices-1):
                                #     z0 = n_slices-1
                                # else:
                                #     z0 = <int> floor(zcof)
                                # z0 = z0 * _magnification_z
                                rgc_map[f, sM, rM, cM] = rgc_val

        return np.asarray(image_interpolated), np.asarray(gradients_r_interpolated), np.asarray(gradients_c_interpolated),np.asarray(gradients_s_interpolated), np.asarray(rgc_map), np.asarray(avg_rgc_map)

    def _run_threaded(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, radius_z: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type="Threaded"):

        interpolator = ShiftAndMagnify()

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius_z / 2.355
        cdef float fwhm_z = radius_z
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
        cdef int w, n_windows, start_frame, end_frame

        cdef float rgc_val, zcof

        for f in range(n_frames):

            # image_interpolated = interpolator.run(image[f], 0, 0, _magnification_xy, _magnification_xy)
            image_interpolated = interpolate_3d(image[f], _magnification_xy, _magnification_z)
            n_slices_mag, n_rows_mag, n_cols_mag = image_interpolated.shape[0], image_interpolated.shape[1], image_interpolated.shape[2]

            gradients_c = np.zeros((n_slices, n_rows*Gx_Gy_MAGNIFICATION, n_cols*Gx_Gy_MAGNIFICATION), dtype=np.float32)
            gradients_r = np.zeros((n_slices, n_rows*Gx_Gy_MAGNIFICATION, n_cols*Gx_Gy_MAGNIFICATION), dtype=np.float32)
            gradients_s = np.zeros((n_slices, n_rows*Gx_Gy_MAGNIFICATION, n_cols*Gx_Gy_MAGNIFICATION), dtype=np.float32)

            img_dum = interpolator.run(image[f], 0, 0, Gx_Gy_MAGNIFICATION, Gx_Gy_MAGNIFICATION)
            # img_dum = interpolate_3d(image[f], Gx_Gy_MAGNIFICATION, _magnification_z)
            n_slices_mag_dum, n_rows_mag_dum, n_cols_mag_dum = img_dum.shape[0], img_dum.shape[1], img_dum.shape[2]

            with nogil:
                _c_gradient_3d(&img_dum[0, 0, 0], &gradients_c[0, 0, 0], &gradients_r[0, 0, 0], &gradients_s[0, 0, 0], n_slices, n_rows_mag_dum, n_cols_mag_dum)

            # gradients_s_interpolated = interpolator.run(gradients_s, 0, 0, _magnification_xy, _magnification_xy)
            # gradients_r_interpolated = interpolator.run(gradients_r, 0, 0, _magnification_xy, _magnification_xy)
            # gradients_c_interpolated = interpolator.run(gradients_c, 0, 0, _magnification_xy, _magnification_xy)

            gradients_s_interpolated = interpolate_3d(gradients_s, _magnification_xy, _magnification_z)
            gradients_r_interpolated = interpolate_3d(gradients_r, _magnification_xy, _magnification_z)
            gradients_c_interpolated = interpolate_3d(gradients_c, _magnification_xy, _magnification_z)

            if self.keep_gradients:
                self._gradients_s_interpolated = gradients_s_interpolated.copy()
                self._gradients_r_interpolated = gradients_r_interpolated.copy()
                self._gradients_c_interpolated = gradients_c_interpolated.copy()

            with nogil:
                for sM in range(0, n_slices_mag):
                    for rM in prange(0, n_rows_mag):
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, Gx_Gy_MAGNIFICATION, 1, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                # zcof = (sM) / _magnification_z
                                # if (zcof < 0):
                                #     z0 = 0
                                # elif (zcof >= n_slices-1):
                                #     z0 = n_slices-1
                                # else:
                                #     z0 = <int> floor(zcof)
                                # z0 = z0 * _magnification_z
                                rgc_map[f, sM, rM, cM] = rgc_val * image_interpolated[sM, rM, cM]
                            else:
                                rgc_val = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, Gx_Gy_MAGNIFICATION, 1, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)
                                # zcof = (sM) / _magnification_z
                                # if (zcof < 0):
                                #     z0 = 0
                                # elif (zcof >= n_slices-1):
                                #     z0 = n_slices-1
                                # else:
                                #     z0 = <int> floor(zcof)
                                # z0 = z0 * _magnification_z
                                rgc_map[f, sM, rM, cM] = rgc_val

        return rgc_map

    def time_analysis(float[:,:,:,:] rgc_map, correlation: str = "AVG", framewindow: float = 10, rollingoverlap: float = 5):
        # correlation (str): Type of correlation to calculate. Should be "AVG", "VAR", or "TAC2"
        n_frames, n_slices, n_rows, n_cols = rgc_map.shape[0], rgc_map.shape[1], rgc_map.shape[2], rgc_map.shape[3]

        if rollingoverlap:
            n_windows = int((n_frames - framewindow) / rollingoverlap) + 1
        else:
            n_windows = int(n_frames / framewindow)       

        cdef float[:, :, :, :] avg_rgc_map = np.zeros((n_windows, n_slices, n_rows, n_cols), dtype=np.float32)

        for w in range(n_windows):
            start_frame = w * (int(framewindow) - int(rollingoverlap))
            end_frame = start_frame + int(framewindow)
            # do average in the first dimension of rgc_map[start_frame:end_frame ,:,:,:] with numpy
            avg_rgc_map[w,:,:,:] = calculate_eSRRF_temporal_correlations(rgc_map[start_frame:end_frame,:,:,:], correlation)

        return avg_rgc_map        

    def get_gradients(self):
        if self._gradients_c_interpolated is None:
            print("Gradients not yet calculated")
        else:
            return self._gradients_c_interpolated, self._gradients_r_interpolated, self._gradients_s_interpolated
