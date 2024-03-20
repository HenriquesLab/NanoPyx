from ...core.transform._le_nlm_denoising import NLMDenoising


def non_local_means_denoising(img: np.ndarray, patch_size: int = 7, patch_distance: int = 11, h: float = 0.1, sigma: float = 0.0):
    denoiser = NLMDenoising()
    return denoiser.run(img, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma)