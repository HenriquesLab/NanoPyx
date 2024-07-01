import os
import tempfile
import pytest
import yaml
import numpy as np
from unittest.mock import MagicMock, patch

from nanopyx.data.download import ExampleDataManager, download, ZipTiffIterator


# Mocking the required modules and functions
def mock_get_examples_path():
    return "/mock/base/path"


def mock_download(url, file_path, download_type=None, unzip=False):
    pass  # Mock download does nothing


class MockZipTiffIterator:
    def __init__(self, file_path):
        pass

    def close(self):
        pass

    def __iter__(self):
        return iter(np.random.rand(10, 10))  # Mock data


@pytest.fixture
def setup_manager():
    with (
        patch("nanopyx.data.download.get_examples_path", mock_get_examples_path),
        patch("nanopyx.data.download.download", mock_download),
        patch("nanopyx.data.download.ZipTiffIterator", MockZipTiffIterator),
    ):
        manager = ExampleDataManager()
        yield manager


def test_initialization(setup_manager):
    manager = setup_manager
    assert manager._temp_dir == os.path.join(tempfile.gettempdir(), "nanopyx_data")


def test_list_datasets(setup_manager):
    manager = setup_manager
    manager._datasets = [{"label": "dataset1", "nickname": "ds1"}, {"label": "dataset2", "nickname": "ds2"}]
    assert manager.list_datasets() == ["dataset1", "dataset2"]


def test_list_datasets_nickname(setup_manager):
    manager = setup_manager
    manager._datasets = [{"label": "dataset1", "nickname": "ds1"}, {"label": "dataset2", "nickname": "ds2"}]
    assert manager.list_datasets_nickname() == [("ds1", "dataset1"), ("ds2", "dataset2")]


def test_get_dataset_info(setup_manager):
    manager = setup_manager
    manager._datasets = [
        {"label": "dataset1", "nickname": "ds1", "info": "info1"},
        {"label": "dataset2", "nickname": "ds2", "info": "info2"},
    ]
    assert manager.get_dataset_info("dataset1") == {"label": "dataset1", "nickname": "ds1", "info": "info1"}
    assert manager.get_dataset_info("ds2") == {"label": "dataset2", "nickname": "ds2", "info": "info2"}
    with pytest.raises(ValueError):
        manager.get_dataset_info("nonexistent")


def test_is_downloaded(setup_manager):
    manager = setup_manager
    manager._datasets = [{"label": "dataset1", "nickname": "ds1", "tiff_sequence_path": "/mock/path/tiff_sequence.zip"}]
    assert manager.is_downloaded("dataset1")


def test_get_thumbnail(setup_manager):
    manager = setup_manager
    manager._datasets = [{"label": "dataset1", "nickname": "ds1", "thumbnail_path": "/mock/path/thumbnail.jpg"}]
    result = manager.get_thumbnail("dataset1")
    assert result == "/mock/path/thumbnail.jpg"


def test_clear_downloads(setup_manager):
    manager = setup_manager
    os.makedirs(manager._temp_dir, exist_ok=True)
    with open(os.path.join(manager._temp_dir, "dummy_file.txt"), "w") as f:
        f.write("dummy content")
    manager.clear_downloads()
    assert not os.path.exists(manager._temp_dir)
