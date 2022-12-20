# adaptation of https://github.com/HenriquesLab/NanoJ-eSRRF/blob/master/src/nanoj/liveSRRF/ErrorMapLiveSRRF.java into NanoPyx

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import (
    brent,  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brent.html
)
from scipy.stats import (
    linregress,  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
)
from skimage.transform import resize

from ..analysis.pearson_correlation import pearsonCorrelation


class ErrorMap:
    def __init__(self):
        self._vRSE: float = 0
        self._vRSP: float = 0
        self._alpha: float = 0
        self._beta: float = 0
        self._sigma: float = 0

        self.imRef: np.ndarray = None
        self.imSR: np.ndarray = None
        self.imSRIntensityScaledBlurred: np.ndarray = None
        self.imRSE: np.ndarray = None

    def optimise(self, imRef: np.ndarray, imSR: np.ndarray, fixedSigma=0) -> None:
        self.imRef = imRef
        self.imSR = imSR

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
                sigmaFunction2Optimize,
                args=(imRef, imSR),
                brack=(0, max_sigma_boundary),
                maxiter=1000,
            )

        if abs(sigma_linear - max_sigma_boundary) < 0.0001:
            print("RSF constrained, as no good minimum found")

        # GET ALPHA AND BETA
        alpha, beta = calculateAlphaBeta(sigma_linear, imRef, imSR)
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma_linear
        self.imSRIntensityScaledBlurred = gaussian_filter(
            imSR * self._alpha + self._beta, self._sigma
        )
        self.imRSE = np.abs(self.imSRIntensityScaledBlurred - imRef)
        self._vRSE = np.mean((self.imSRIntensityScaledBlurred - imRef) ** 2) ** 0.5
        self._vRSP = pearsonCorrelation(self.imSRIntensityScaledBlurred, imRef)

    def getRSE(self) -> float:
        return self._vRSE

    def getRSP(self) -> float:
        return self._vRSP

    def getSigma(self) -> float:
        return self._sigma


def calculateAlphaBeta(
    sigma: float, imRef: np.ndarray, imSR: np.ndarray
) -> tuple[float, float]:
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


def sigmaFunction2Optimize(sigma: float, imRef: np.ndarray, imSR: np.ndarray) -> float:
    alpha, beta = calculateAlphaBeta(sigma, imRef, imSR)
    imSRIntensityScaledBlurred = gaussian_filter(imSR * alpha + beta, sigma)
    rmse = np.mean((imSRIntensityScaledBlurred - imRef) ** 2) ** 0.5
    return rmse
