from nanopyx.core.transform import FFT

def test_fft(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True)

    lefft = FFT(clear_benchmarks=True, testing=True)
    lefft.benchmark(dataset)