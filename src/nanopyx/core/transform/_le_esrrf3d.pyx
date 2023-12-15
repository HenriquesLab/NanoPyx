import numpy as np

cimport numpy as np

from ._interpolation import interpolate_3d
from ...__liquid_engine__ import LiquidEngine

cdef extern from "_c_gradients.h":
    void _c_gradient_3d(float* pixels, float* GcArray, float* GrArray, float* GsArray, int frame, int w, int h) nogil

cdef extern from "_c_sr_radial_gradient_convergence.h":
    float _c_calculate_rgc3D(float xM, float yM, float sliceM, float* imIntGx, float* imIntGy, float* imIntGz, int colsM, int rowsM, int slicesM, int _magnification_xy, int _magnification_z, float Gx_Gy_MAGNIFICATION, float Gz_MAGNIFICATION, float fwhm, float fwhm_z, float tSO, float tSO_z, float tSS, float tSS_z, float sensitivity) nogil

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

    def run(self, image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type=None):
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if len(image.shape) == 4:
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)
        elif len(image.shape) == 3:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            return self._run(image, magnification_xy=magnification_xy, magnification_z=magnification_z, radius=radius, sensitivity=sensitivity, doIntensityWeighting=doIntensityWeighting, run_type=run_type)
    
    def _run_unthreaded(self, float[:,:,:,:] image, magnification_xy: int = 5, magnification_z: int = 5, radius: float = 1.5, radius_z: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, run_type="Unthreaded"):

        cdef float sigma = radius / 2.355
        cdef float fwhm = radius
        cdef float tSS = 2 * sigma * sigma
        cdef float tSO = 2 * sigma + 1
        cdef float sigma_z = radius_z / 2.355
        cdef float fwhm_z = radius_z
        cdef float tSS_z = 2 * sigma_z * sigma_z
        cdef float tSO_z = 2 * sigma_z + 1
        cdef int Gx_Gy_Gz_MAGNIFICATION = 2
        cdef int _magnification_xy = magnification_xy
        cdef int _magnification_z = magnification_z
        cdef int _doIntensityWeighting = doIntensityWeighting

        cdef int n_frames, n_slices, n_rows, n_cols
        n_frames, n_slices, n_rows, n_cols = image.shape[0], image.shape[1], image.shape[2], image.shape[3]

        cdef float[:, :, :, :] rgc_map = np.zeros((n_frames, n_slices*magnification_z, n_rows*magnification_xy, n_cols*magnification_xy), dtype=np.float32)
        cdef float[:, :, :] image_interpolated, gradients_s, gradients_r, gradients_c, gradients_s_interpolated, gradients_r_interpolated, gradients_c_interpolated, padded

        cdef int f, n_slices_mag, n_rows_mag, n_cols_mag, sM, rM, cM

        for f in range(n_frames):

            image_interpolated = interpolate_3d(image[f], _magnification_xy, _magnification_z)
            n_slices_mag, n_rows_mag, n_cols_mag = image_interpolated.shape[0], image_interpolated.shape[1], image_interpolated.shape[2]

            gradients_c = np.zeros((n_slices*Gx_Gy_Gz_MAGNIFICATION+1, n_rows*Gx_Gy_Gz_MAGNIFICATION+1, n_cols*Gx_Gy_Gz_MAGNIFICATION+1), dtype=np.float32)
            gradients_r = np.zeros((n_slices*Gx_Gy_Gz_MAGNIFICATION+1, n_rows*Gx_Gy_Gz_MAGNIFICATION+1, n_cols*Gx_Gy_Gz_MAGNIFICATION+1), dtype=np.float32)
            gradients_s = np.zeros((n_slices*Gx_Gy_Gz_MAGNIFICATION+1, n_rows*Gx_Gy_Gz_MAGNIFICATION+1, n_cols*Gx_Gy_Gz_MAGNIFICATION+1), dtype=np.float32)

            padded = np.pad(interpolate_3d(image[f], Gx_Gy_Gz_MAGNIFICATION, Gx_Gy_Gz_MAGNIFICATION), ((0,1), (0,1), (0,1)), 'edge')

            with nogil:
                _c_gradient_3d(&padded[0, 0, 0], &gradients_c[0, 0, 0], &gradients_r[0, 0, 0], &gradients_s[0, 0, 0], n_slices*Gx_Gy_Gz_MAGNIFICATION+1, n_rows*Gx_Gy_Gz_MAGNIFICATION+1, n_cols*Gx_Gy_Gz_MAGNIFICATION+1)

            gradients_s_interpolated = interpolate_3d(gradients_s[0:n_slices*Gx_Gy_Gz_MAGNIFICATION, 0:n_rows*Gx_Gy_Gz_MAGNIFICATION , 0:n_cols*Gx_Gy_Gz_MAGNIFICATION].copy(), _magnification_xy, _magnification_z)
            gradients_r_interpolated = interpolate_3d(gradients_r[0:n_slices*Gx_Gy_Gz_MAGNIFICATION, 0:n_rows*Gx_Gy_Gz_MAGNIFICATION , 0:n_cols*Gx_Gy_Gz_MAGNIFICATION].copy(), _magnification_xy, _magnification_z)
            gradients_c_interpolated = interpolate_3d(gradients_c[0:n_slices*Gx_Gy_Gz_MAGNIFICATION, 0:n_rows*Gx_Gy_Gz_MAGNIFICATION , 0:n_cols*Gx_Gy_Gz_MAGNIFICATION].copy(), _magnification_xy, _magnification_z)

            with nogil:
                for sM in range(0, n_slices_mag):
                    for rM in range(0, n_rows_mag):
                        for cM in range(0, n_cols_mag):
                            if _doIntensityWeighting:
                                rgc_map[f, sM, rM, cM] = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, Gx_Gy_Gz_MAGNIFICATION, Gx_Gy_Gz_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity) * image_interpolated[sM, rM, cM]
                            else:
                                rgc_map[f, sM, rM, cM] = _c_calculate_rgc3D(cM, rM, sM, &gradients_c_interpolated[0,0,0], &gradients_r_interpolated[0,0,0], &gradients_s_interpolated[0,0,0], n_cols_mag, n_rows_mag, n_slices_mag, _magnification_xy, _magnification_z, Gx_Gy_Gz_MAGNIFICATION, Gx_Gy_Gz_MAGNIFICATION, fwhm, fwhm_z, tSO, tSO_z, tSS, tSS_z, sensitivity)

        return np.asarray(padded), np.asarray(gradients_r), np.asarray(gradients_c),np.asarray(gradients_s), np.asarray(rgc_map)

