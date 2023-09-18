# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

# adaptation of https://github.com/HenriquesLab/NanoJ-eSRRF/blob/master/src/nanoj/liveSRRF/ErrorMapLiveSRRF.java into NanoPyx

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import (
    brent,  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html
)
from scipy.stats import (
    linregress,  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
)

# TODO RECHECK VALUES
from ._le_interpolation_catmull_rom import ShiftAndMagnify

from ..analysis.pearson_correlation import pearson_correlation


cdef class ErrorMap:
    cdef public float _vRSE, _vRSP, _alpha, _beta, _sigma
    cdef public float[:, :] imRSE, img_ref, img_sr, im_sr_intensity_scaled_blurred

    def __init__(self):

        self._vRSE: float = 0
        self._vRSP: float = 0
        self._alpha: float = 0
        self._beta: float = 0
        self._sigma: float = 0

    def optimise(self, img_ref: np.ndarray, img_sr: np.ndarray, fixedSigma=0) -> None:
        
        self._optimise(img_ref.astype(np.float32), img_sr.astype(np.float32), fixedSigma)


    cdef void _optimise(self, float[:,:] img_ref, float[:,:] img_sr, int fixedSigma):

        self.img_ref = img_ref
        self.img_sr = img_sr

        cdef object interpolator = ShiftAndMagnify()

        cdef int magnification = <int>(img_sr.shape[0] / img_ref.shape[0])
        assert magnification == int(img_sr.shape[1] / img_ref.shape[1])
        
        
        cdef float[:, :] img_ref_int = np.zeros((np.asarray(img_ref).shape[0]*magnification, np.asarray(img_ref).shape[1]*magnification)).astype(np.float32)
        #cdef float[:, :] imRSE = np.
        
        if magnification > 1:
            img_ref_int[:,:] = interpolator.run(np.asarray(img_ref).astype(np.float32),0,0,magnification,magnification) 
            #imRef = resize(imRef, imSR.shape, order=3, preserve_range=True)

        #self.img_ref_magnified = img_ref_int

        max_sigma_boundary = (
            4 / 2.35482
        ) * magnification  # this assumes Nyquist sampling in the ref image

        sigma_linear = fixedSigma * magnification
        if fixedSigma == 0:
            sigma_linear = brent(
                sigma_function_to_optimize,
                args=(img_ref_int, img_sr),
                brack=(0, max_sigma_boundary),
                maxiter=1000,
            )

        if abs(sigma_linear - max_sigma_boundary) < 0.0001:
            print("RSF constrained, as no good minimum found")

        # GET ALPHA AND BETA
        alpha, beta = calculate_alpha_beta(sigma_linear, img_ref_int, img_sr)
        #alpha, beta = self._calculate_alpha_beta(sigma_linear, img_ref_int, img_sr)
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma_linear
        self.im_sr_intensity_scaled_blurred = gaussian_filter(
            self._alpha * np.asarray(img_sr) + self._beta, self._sigma
        )
        self.imRSE = np.abs(np.asarray(self.im_sr_intensity_scaled_blurred) - np.asarray(img_ref_int))
        self._vRSE = np.mean((np.asarray(self.im_sr_intensity_scaled_blurred) - np.asarray(img_ref_int)) ** 2) ** 0.5
        self._vRSP = pearson_correlation(np.asarray(self.im_sr_intensity_scaled_blurred), np.asarray(img_ref_int))

    def getRSE(self) -> float:
        return self._vRSE

    def getRSP(self) -> float:
        return self._vRSP

    def get_sigma(self) -> float:
        return self._sigma


# cdef float[:, :] _calculate_alpha_beta(float sigma, float[:,:] img_ref, float[:,:] img_sr):

#     cdef float[:, :] img_sr_blurred 
#     cdef float slope, intercept, r, p, se 

#     img_sr_blurred = gaussian_filter(img_sr, sigma)
#     slope, intercept, r, p, se = linregress(np.asarray(img_sr_blurred).ravel(), np.asarray(img_ref).ravel())

#     return slope, intercept


# cdef float _sigma_function_to_optimize(float sigma, float[:,:] img_ref, float[:,:] img_sr):
#     cdef float alpha, beta
#     cdef float[:,:] im_sr_intensity_scaled_blurred

#     alpha, beta = _calculate_alpha_beta(sigma, img_ref, img_sr)
#     im_sr_intensity_scaled_blurred = gaussian_filter(np.asarray(img_sr) * alpha + beta, sigma)

#     rmse = np.mean((np.asarray(im_sr_intensity_scaled_blurred) - np.asarray(img_ref)) ** 2) ** 0.5
#     return rmse

# def sigma_function_to_optimize(sigma: float, img_ref: np.ndarray, img_sr: np.ndarray):
#     return _sigma_function_to_optimize(sigma, img_ref, img_sr)

# def calculate_alpha_beta(sigma: float, img_ref: np.ndarray, img_sr: np.ndarray):
#     return _calculate_alpha_beta(sigma, img_ref, img_sr)


def calculate_alpha_beta(
    sigma: float, imRef: np.ndarray, imSR: np.ndarray
) -> tuple:
    """Gaussian blurs imSR image and calculates linear regressino again imRef

    Args:
        sigma (float): gaussian blur sigma
        imRef (np.ndarray): reference image (generally a difraction limited equivalent)
        imSR (np.ndarray): super-resolution image

    Returns:
        tuple[float, float]: alpha and beta for linear regression
    """
    imSRBlurred = gaussian_filter(imSR, sigma)
    slope, intercept, r, p, se = linregress(np.asarray(imSRBlurred).ravel(), np.asarray(imRef).ravel())
    return slope, intercept

def sigma_function_to_optimize(sigma: float, imRef: np.ndarray, imSR: np.ndarray) -> float:
    alpha, beta = calculate_alpha_beta(sigma, imRef, imSR)
    im_sr_intensity_scaled_blurred = gaussian_filter(imSR * alpha + beta, sigma)
    rmse = np.mean((im_sr_intensity_scaled_blurred - imRef) ** 2) ** 0.5
    return rmse
