from nanopyx.core.transform.sr_radial_gradient_convergence import RadialGradientConvergence
from nanopyx.data.download import ExampleDataManager
from matplotlib import pyplot as plt
import numpy as np

#downloader = ExampleDataManager()

def test_rgc(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True)
    rgc = RadialGradientConvergence(sensitivity=2)

    small_dataset = dataset[:10,:20,:20]

    imRad, imInt, imGx, imGy = rgc.calculate(small_dataset)
    plt.imshow(np.mean(imRad,0), interpolation='none')
    plt.title('RGC')
    #plt.show()

#test_rgc(downloader)