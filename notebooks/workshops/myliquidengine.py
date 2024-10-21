import numpy as np
from liquid_engine import LiquidEngine

from skimage.restoration import denoise_nl_means

class MyLiquidEngineClass(LiquidEngine):
    def __init__(self):
        self._designation = "MyLiquidEngineClass"
        super().__init__()

    def run(self, image: np.ndarray, patch_size: int, patch_distance: int, h:float, sigma:float, run_type=None):
        return self._run(image, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma)

    def _run_ski_nlm_1(self, image, patch_size, patch_distance, h, sigma):
        return denoise_nl_means(image, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma, fast_mode=True)

    def _run_ski_nlm_2(self, image, patch_size, patch_distance, h, sigma):
        return denoise_nl_means(image, patch_size=patch_size, patch_distance=patch_distance, h=h, sigma=sigma, fast_mode=False)