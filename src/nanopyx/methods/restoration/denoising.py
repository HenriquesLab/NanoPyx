import numpy as np
from ...core.transform._le_nlm_denoising import NLMDenoising


def non_local_means_denoising(img: np.ndarray, patch_size: int = 7, patch_distance: int = 11, h: float = 0.1, sigma: float = 0.0):
    """
    Apply Non-Local Means (NLM) denoising algorithm to an image.

    Parameters
    ----------
    img : np.ndarray
        The input image as a 2D numpy array.
    patch_size : int, optional
        The size of the square patch used for denoising. Default is 7.
    patch_distance : int, optional
        The maximum distance between any two patches used for denoising. Default is 11.
    h : float, optional
        The filtering parameter controlling the degree of smoothing. Higher values increase smoothing. Default is 0.1.
    sigma : float, optional
        The standard deviation of the noise (if known). Default is 0.0, which means it is estimated from the image.

    Returns
    -------
    np.ndarray
        The denoised image as a 2D numpy array.

    Notes
    -----
    The Non-Local Means algorithm denoises an image by replacing each pixel's value with an average of similar pixels in a local neighborhood. This method is particularly effective for preserving edges and fine details in images.
    """
    denoiser = NLMDenoising()
    return denoiser.run(img, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma)
