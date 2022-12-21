import os
import shutil
import tempfile

import pkg_resources
import wget
import yaml
from onedrivedownloader import download as onedrive_download

from .checksum import getChecksum
from .zip_image_loader import open_tiffs_in_zip


class GetExampleData:
    _file_name = "examples.yaml"
    _temp_dir = os.path.join(tempfile.gettempdir(), "nanopix_data")
    _dataset_files = {}

    def __init__(self, path_example_yaml: str = None):

        if path_example_yaml is None:
            possible_paths = [
                pkg_resources.resource_filename(
                    "nanopyx", os.path.join("data", self._file_name)
                ),
                os.path.join(os.getcwd(), self._file_name),
                os.path.join(
                    os.sep.join(os.path.split(__file__)[:-1]), self._file_name
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    path_example_yaml = path

            if path_example_yaml is None:
                raise ValueError(f"Could not find path to file {self._file_name}")

        # Open the YAML file
        with open(path_example_yaml, "r") as f:
            # Load the YAML contents
            self.data = yaml.load(f, Loader=yaml.FullLoader)

    def list_datasets(self):
        return self.data["datasets"]

    def download_dataset(
        self, label: str, path: str = None, return_dataset: bool = True
    ):
        if path is None:
            if not os.path.exists(self._temp_dir):
                os.mkdir(self._temp_dir)
            path = self._temp_dir
        else:
            assert os.path.exists(path)

        dataset = self.data["datasets"][label]
        file_path = os.path.join(path, dataset["filename"])

        # clear wrong file
        if os.path.exists(file_path) and getChecksum(file_path) != dataset["checksum"]:
            os.remove(file_path)

        if not os.path.exists(file_path):
            if dataset["url_type"] == "onedrive":
                onedrive_download(
                    dataset["url"], file_path, unzip=False
                )  # , unzip=False, unzip_path: str = None, force_download=False, force_unzip=False, clean=False)
            else:
                wget.download(url=dataset["url"], out=file_path)

        self._dataset_files = {label: file_path}
        if return_dataset:
            if dataset["unpacking"]["format"] == "zip":
                return open_tiffs_in_zip(file_path, dataset["unpacking"]["pattern"])

    def clear_downloads(self):
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
