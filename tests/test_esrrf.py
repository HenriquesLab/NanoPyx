from nanopyx.core.transform import RadialGradientConvergence 
from nanopyx.core.transform import CRShiftAndMagnify, GradientRobertsCross
from nanopyx.methods import eSRRF

def test_rgc(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True)
    small_dataset = dataset[:10,:20,:20]

    crsm = CRShiftAndMagnify()
    grc = GradientRobertsCross(testing=True,clear_benchmarks=True)
    rgc = RadialGradientConvergence(testing=True,clear_benchmarks=True)
    
    small_dataset_interp = crsm.run(small_dataset, 0, 0, 5, 5)
    gradient_col, gradient_row = grc.run(small_dataset)
    gradient_col_interp = crsm.run(gradient_col, 0, 0, 5*2, 5*2)
    gradient_row_interp = crsm.run(gradient_row, 0, 0, 5*2, 5*2)
    radial_gradients = rgc.benchmark(gradient_col_interp, gradient_row_interp, small_dataset_interp)

def test_esrrf(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True)
    small_dataset = dataset[:10,:20,:20]
    
    eSRRF(small_dataset)