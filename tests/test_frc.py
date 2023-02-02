import numpy as np
from nanopyx.data.download import ExampleDataManager
from nanopyx.core.sr.frc import FIRECalculator

downloader = ExampleDataManager()

img = downloader.get_ZipTiffIterator("SMLMS2013_HDTubulinAlexa647", as_ndarray = True)
img_1 = np.sum(img[:250], axis=0)
img_2 = np.sum(img[250:], axis=0)

fire_calculator = FIRECalculator()
fire = fire_calculator.calculate_fire_number(img_1, img_2, "Fixed 1/7")

