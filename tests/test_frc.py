import numpy as np
from nanopyx.data.download import ExampleDataManager
from nanopyx.core.sr.frc import FIRECalculator, calculate_FIRE

downloader = ExampleDataManager()

img = downloader.get_ZipTiffIterator("SMLMS2013_HDTubulinAlexa647", as_ndarray = True)
img_1 = np.sum(img[:250], axis=0)
img_2 = np.sum(img[250:], axis=0)

fire_calculator = FIRECalculator()
fire_fixed = fire_calculator.calculate_fire_number(img_1, img_2, "Fixed 1/7")
fire_hb = fire_calculator.calculate_fire_number(img_1, img_2, "Half-bit")
fire_3s = fire_calculator.calculate_fire_number(img_1, img_2, "Three sigma")

fire = calculate_FIRE(img_1, img_2, pixel_recon_dim=100)