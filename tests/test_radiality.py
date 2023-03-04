from nanopyx.core.transform.sr_radiality import Radiality


def test_radiality(downloader, plt):
    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )

    r = Radiality(magnification=2, ringRadius=0.5)
    imRad, imIW, imGx, imGy = r.calculate(dataset)

    plt.imshow(imRad.mean(0), interpolation="none")
    plt.title("Radiality")
