from nanopyx.data.download import ExampleDataManager
from nanopyx.core.io.zip_image_loader import ZipTiffIterator


def test_list_datasets(downloader: ExampleDataManager):
    datasets = downloader.list_datasets()
    assert isinstance(datasets, list)
    assert all(isinstance(d, str) for d in datasets)


def test_list_datasets_nickname(downloader: ExampleDataManager):
    datasets = downloader.list_datasets_nickname()
    assert isinstance(datasets, list)
    assert all(isinstance(d, tuple) and len(d) == 2 for d in datasets)
    assert all(isinstance(d[0], str) and isinstance(d[1], str) for d in datasets)


def test_get_dataset_info(downloader: ExampleDataManager):
    dataset_info = downloader.get_dataset_info("SMLMS2013_HDTubulinAlexa647")
    assert isinstance(dataset_info, dict)


def test_load_dataset(downloader: ExampleDataManager):
    tiff_iterator = downloader.get_ZipTiffIterator(
        "SMLMS2013_HDTubulinAlexa647", as_ndarray=False
    )
    assert isinstance(tiff_iterator, ZipTiffIterator)
