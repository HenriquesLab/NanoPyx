import numpy as np
from nanopyx.__liquid_engine__ import LiquidEngine
from skimage.restoration import denoise_nl_means


class MyLiquidEngineClass(LiquidEngine):

    def __init__(self, clear_benchmarks=False, testing=False, verbose=True):
        self._designation = "MyLiquidEngineClass"
        super().__init__(
            clear_benchmarks=clear_benchmarks, testing=testing, verbose=verbose)

    def run(self, image: np.ndarray, patch_size: int, patch_distance: int, h: float, sigma: float, run_type:bool=None):
        if image.dtype != "np.float32":
            image = image.astype("np.float32")
        return self._run(image, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma)

    def _run_ski_nlm_fast(self, image, patch_size, patch_distance, h, sigma):
        return denoise_nl_means(image, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma, fast_mode=True)

    def _run_ski_nlm_nonfast(self, image, patch_size, patch_distance, h, sigma):
        return denoise_nl_means(image, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma, fast_mode=False)