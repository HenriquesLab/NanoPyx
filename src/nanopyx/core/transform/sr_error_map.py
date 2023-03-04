# adaptation of https://github.com/HenriquesLab/NanoJ-eSRRF/blob/master/src/nanoj/liveSRRF/ErrorMapLiveSRRF.java

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import (
    brent,  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html
)
from scipy.stats import (
    linregress,  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
)
from skimage.transform import resize

from ..analysis.pearson_correlation import pearson_correlation


class ErrorMap:
    def __init__(self):
        self._vRSE: float = 0
        self._vRSP: float = 0
        self._alpha: float = 0
        self._beta: float = 0
        self._sigma: float = 0

        self.im_ref: np.ndarray = None
        self.im_sr: np.ndarray = None
        self.im_sr_intensity_scaled_blurred: np.ndarray = None
        self.imRSE: np.ndarray = None

    def optimise(self, imRef: np.ndarray, imSR: np.ndarray, fixedSigma=0) -> None:
        self.im_ref = imRef
        self.im_sr = imSR

        magnification = imSR.shape[0] / imRef.shape[0]
        assert magnification == imSR.shape[1] / imRef.shape[1]

        if magnification > 1:
            imRef = resize(imRef, imSR.shape, order=3, preserve_range=True)

        self.imRefMagnified = imRef

        max_sigma_boundary = (
            4 / 2.35482
        ) * magnification  # this assumes Nyquist sampling in the ref image

        sigma_linear = fixedSigma * magnification
        if fixedSigma == 0:
            sigma_linear = brent(
                sigma_function_to_optimize,
                args=(imRef, imSR),
                brack=(0, max_sigma_boundary),
                maxiter=1000,
            )

        if abs(sigma_linear - max_sigma_boundary) < 0.0001:
            print("RSF constrained, as no good minimum found")

        # GET ALPHA AND BETA
        alpha, beta = calculate_alpha_beta(sigma_linear, imRef, imSR)
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma_linear
        self.im_sr_intensity_scaled_blurred = gaussian_filter(
            imSR * self._alpha + self._beta, self._sigma
        )
        self.imRSE = np.abs(self.im_sr_intensity_scaled_blurred - imRef)
        self._vRSE = np.mean((self.im_sr_intensity_scaled_blurred - imRef) ** 2) ** 0.5
        self._vRSP = pearson_correlation(self.im_sr_intensity_scaled_blurred, imRef)

    def getRSE(self) -> float:
        return self._vRSE

    def getRSP(self) -> float:
        return self._vRSP

    def get_sigma(self) -> float:
        return self._sigma


def calculate_alpha_beta(sigma: float, imRef: np.ndarray, imSR: np.ndarray) -> tuple:
    """Gaussian blurs imSR image and calculates linear regressino again imRef

    Args:
        sigma (float): gaussian blur sigma
        imRef (np.ndarray): reference image (generally a difraction limited equivalent)
        imSR (np.ndarray): super-resolution image

    Returns:
        tuple[float, float]: alpha and beta for linear regression
    """
    imSRBlurred = gaussian_filter(imSR, sigma)
    slope, intercept, r, p, se = linregress(imSRBlurred.ravel(), imRef.ravel())
    return slope, intercept


def sigma_function_to_optimize(
    sigma: float, imRef: np.ndarray, imSR: np.ndarray
) -> float:
    alpha, beta = calculate_alpha_beta(sigma, imRef, imSR)
    im_sr_intensity_scaled_blurred = gaussian_filter(imSR * alpha + beta, sigma)
    rmse = np.mean((im_sr_intensity_scaled_blurred - imRef) ** 2) ** 0.5
    return rmse
