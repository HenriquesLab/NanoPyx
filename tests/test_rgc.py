from nanopyx.core.sr.radial_gradient_convergence import RadialGradientConvergence
from nanopyx.data.download import ExampleDataManager
from matplotlib import pyplot as plt
import numpy as np

#downloader = ExampleDataManager()

def test_rgc(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True)
    rgc = RadialGradientConvergence(sensitivity=2)

    imRad, imInt, imGx, imGy = rgc.calculate(dataset)
    plt.imshow(np.mean(imRad,0), interpolation='none')
    plt.title('RGC')
    #plt.show()

#test_rgc(downloader)