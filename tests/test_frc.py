import numpy as np
from nanopyx.data.download import ExampleDataManager
from nanopyx.core.analysis.frc import FIRECalculator

downloader = ExampleDataManager()

img = downloader.get_ZipTiffIterator("SMLMS2013_HDTubulinAlexa647", as_ndarray = True)

calculator = FIRECalculator(pixel_size=100, units="nm")
fire = calculator.calculate_fire_number(img[0], img[50])