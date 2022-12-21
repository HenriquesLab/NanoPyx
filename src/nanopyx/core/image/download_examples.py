import os
import shutil
import pkg_resources
import yaml
import tempfile
import wget
from urllib.request import ProxyHandler, build_opener, install_opener, urlretrieve
from ..utils.download_file import DownloadFile

from onedrivedownloader import download as onedrive_download

class GetExampleData:
    _file_name = "examples.yaml"
    _temp_dir = os.path.join(tempfile.gettempdir(), "nanopix_data")

    def __init__(self, path_example_yaml: str = None):

        proxy = ProxyHandler({})
        opener = build_opener(proxy)
        opener.addheaders = [
            (
                "User-Agent",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30",
            )
        ]
        install_opener(opener)

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

    def download_dataset(self, label: str, path: str = None):
        if path is None:
            if not os.path.exists(self._temp_dir):
                os.mkdir(self._temp_dir)
            path = self._temp_dir
        else:
            assert os.path.exists(path)

        dataset = self.data["datasets"][label]
        file_path = os.path.join(path, dataset["filename"])
        if os.path.exists(file_path):
            if os.path.getsize(file_path) != dataset["filesize"]:
                os.remove(file_path)
            else:
                return file_path

        if dataset["url_type"] == "onedrive":
            onedrive_download(dataset["url"], file_path, unzip=False)  # , unzip=False, unzip_path: str = None, force_download=False, force_unzip=False, clean=False)
        else:
            wget.download(url=dataset["url"], out=file_path)

        return file_path

    def clear_downloads(self):
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
