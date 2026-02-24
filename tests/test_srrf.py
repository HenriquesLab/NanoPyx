from nanopyx.core.transform import Radiality, CRShiftAndMagnify
from nanopyx.methods import SRRF
from nanopyx.data.download import ExampleDataManager


def test_radiality():
    downloader = ExampleDataManager()
    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    interp = CRShiftAndMagnify()
    small_dataset_interp = interp.run(small_dataset, 0, 0, 5, 5)

    liquid_rad = Radiality(testing=True, clear_benchmarks=True)
    imRad = liquid_rad.benchmark(small_dataset, small_dataset_interp)


def test_srrf():
    downloader = ExampleDataManager()
    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    SRRF(small_dataset)
