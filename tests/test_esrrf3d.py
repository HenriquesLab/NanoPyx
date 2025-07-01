import numpy as np
from nanopyx.methods import eSRRF3D as eSRRF3D_w
from nanopyx.core.transform import eSRRF3D_ST


def test_esrrf3d_workflow(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    small_dataset = small_dataset[np.newaxis, ...]

    eSRRF3D_w(small_dataset)


def test_esrrf3d_mag11(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    dataset = dataset[np.newaxis, ...]

    e3d = eSRRF3D_ST(testing=True, clear_benchmarks=True)

    e3d.benchmark(small_dataset, magnification_xy=1, magnification_z=1)


def test_esrrf3d_mag12(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    dataset = dataset[np.newaxis, ...]

    e3d = eSRRF3D_ST(testing=True, clear_benchmarks=True)

    e3d.benchmark(small_dataset, magnification_xy=1, magnification_z=2)


def test_esrrf3d_mag21(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    dataset = dataset[np.newaxis, ...]

    e3d = eSRRF3D_ST(testing=True, clear_benchmarks=True)

    e3d.benchmark(small_dataset, magnification_xy=2, magnification_z=1)


def test_esrrf3d_mag22(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    dataset = dataset[np.newaxis, ...]

    e3d = eSRRF3D_ST(testing=True, clear_benchmarks=True)

    e3d.benchmark(small_dataset, magnification_xy=2, magnification_z=2)


def test_esrrf3d_radius_too_high(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    dataset = dataset[np.newaxis, ...]

    e3d = eSRRF3D_ST(testing=True, clear_benchmarks=True)

    assert e3d.run(small_dataset, radius=15) is None


def test_esrrf3d_radiusz_too_high(downloader):

    dataset = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=True
    )
    small_dataset = dataset[:10, :20, :20]

    dataset = dataset[np.newaxis, ...]

    e3d = eSRRF3D_ST(testing=True, clear_benchmarks=True)

    assert e3d.run(small_dataset, PSF_ratio=15) is None
