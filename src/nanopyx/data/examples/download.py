import os
import shutil
import tempfile
from urllib.request import ProxyHandler, build_opener, install_opener

import numpy as np
import wget
import yaml
from onedrivedownloader import download as onedrive_download

from nanopyx.core.io.zip_image_loader import ZipTiffIterator

from ...core.io.checksum import getChecksum


class ExampleDataManager:
    _base_bath = os.path.split(__file__)[0]
    _temp_dir = os.path.join(tempfile.gettempdir(), "nanopix_data")
    _to_download_path = None

    def __init__(self, to_download_path: str = None):
        """
        If to_download_path is None, a temporary directory will be created.
        Note that it will not be automatically deleted.
        To clear downloads use self._clear_download()
        """

        # Set download path
        if to_download_path is None:
            self._to_download_path = self._temp_dir
        else:
            self._to_download_path = to_download_path

        # Lets check on how many examples we have available
        self._datasets = []
        for path in os.listdir(self._base_bath):
            full_path = os.path.join(self._base_bath, path)
            info_file_path = os.path.join(full_path, "info.yaml")
            if os.path.isdir(full_path) and os.path.exists(info_file_path):
                info_data = None
                with open(os.path.join(info_file_path), "r") as f:
                    # Load the YAML contents
                    info_data = yaml.load(f, Loader=yaml.FullLoader)

                info = {
                    "info_path": info_file_path,
                    "thumbnail_path": os.path.join(
                        self._base_bath, path, "thumbnail.jpg"
                    ),
                    "tiff_sequence_path": None,
                    "zarr_path": None,
                }
                zarr_path = os.path.join(
                    self._to_download_path, path, "dataset.zarr.zip"
                )
                if os.path.exists(zarr_path):
                    info["zarr_path"] = zarr_path
                tiff_sequence_path = os.path.join(
                    self._to_download_path, path, "tiff_sequence.zip"
                )
                if os.path.exists(tiff_sequence_path):
                    info["tiff_sequence_path"] = tiff_sequence_path

                for key in info_data:
                    info[key] = info_data[key]

                self._datasets.append(info)

        # Fix agent
        proxy = ProxyHandler({})
        opener = build_opener(proxy)
        opener.addheaders = [
            (
                "User-Agent",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30",
            )
        ]
        install_opener(opener)

    def list_datasets(self) -> tuple[list]:
        """
        Returns a list of dataset labels
        """
        return [dataset["label"] for dataset in self._datasets]

    def list_datasets_nickname(self) -> tuple[list]:
        """
        Returns a list of dataset labels
        """
        return [(dataset["nickname"], dataset["label"]) for dataset in self._datasets]

    def get_dataset_info(self, dataset_name: str) -> dict:
        """
        Returns a dictionary with information about the dataset

        Parameters
            dataset_name (str): can be a dataset label or nickname
        """
        for dataset in self._datasets:
            if dataset_name in (dataset["label"], dataset["nickname"]):
                return dataset
        raise ValueError(f"{dataset_name} not found in example datasets")

    def _download(self, url, file_path, download_type=None, unzip=False):
        if os.path.exists(
            file_path
        ):  # or os.path.exists(os.path.splitext(file_path)[0]):
            # raise Warning(f"already exists, no need to download: {file_path}")
            return

        if not os.path.exists(self._temp_dir):
            os.mkdir(self._temp_dir)

        base_path = os.path.split(file_path)[0]
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        if download_type == "onedrive":
            onedrive_download(url, file_path, unzip=unzip, clean=True)
        else:
            wget.download(url=url, out=file_path)

    def _copy_auxiliary_files(self, info: dict):
        if not os.path.exists(self._to_download_path):
            os.mkdir(self._to_download_path)

        path = os.path.join(self._to_download_path, info["label"])
        if not os.path.exists(path):
            os.mkdir(path)

        thumbnail_path = os.path.join(path, "thumbnail.jpg")
        if not os.path.exists(thumbnail_path):
            shutil.copyfile(info["thumbnail_path"], thumbnail_path)

        info_path = os.path.join(path, "info.yaml")
        if not os.path.exists(info_path):
            shutil.copyfile(info["info_path"], info_path)

    def download_zarr(self, dataset_name: str) -> str:
        """
        Downloads the zarr dataset and returns the path to the zarr file
        """

        info = self.get_dataset_info(dataset_name)
        path = os.path.join(self._to_download_path, info["label"])

        file_path_zip = os.path.join(path, "dataset.zarr.zip")
        file_path_zarr = os.path.join(path, "dataset.zarr")
        if os.path.exists(file_path_zarr):
            return file_path_zarr

        url = info["zarr_url"]
        download_type = info["zarr_url_type"]
        self._copy_auxiliary_files(info)
        self._download(url, file_path_zip, download_type, unzip=True)
        info["zarr_path"] = file_path_zarr

        return file_path_zarr

    def download_tiff_sequence(self, dataset_name: str) -> str:
        """
        Downloads the tiff sequence and returns the path to the zip file
        """
        info = self.get_dataset_info(dataset_name)
        path = os.path.join(self._to_download_path, info["label"])

        file_path = os.path.join(path, "tiff_sequence.zip")
        url = info["tiff_sequence_url"]
        download_type = info["tiff_sequence_url_type"]

        self._copy_auxiliary_files(info)
        self._download(url, file_path, download_type)
        info["tiff_sequence_path"] = file_path

        return file_path

    # def get_zarr(self, dataset_name: str) -> zarr.hierarchy.Group:
    #     """
    #     Downloads the zarr dataset and returns the zarr group
    #     """
    #     self._show_citation_notice(dataset_name)
    #     file_path = self.download_zarr(dataset_name)
    #     z = zarr.open(file_path, mode="r")
    #     return z

    def get_ZipTiffIterator(self, dataset_name: str, as_ndarray: False) -> ZipTiffIterator:
        """
        Downloads the tiff sequence and returns the ZipTiffIterator

        Parameters
            dataset_name (str): can be a dataset label or nickname
            as_ndarray (bool): if True, returns a numpy array instead of a ZipTiffIterator
        """
        self._show_citation_notice(dataset_name)
        file_path = self.download_tiff_sequence(dataset_name)
        zti = ZipTiffIterator(file_path)
        if not as_ndarray:
            return zti      
        else:
            arr = np.asarray(zti)
            zti.close()
            return arr

    def get_thumbnail(self, dataset_name: str) -> str:
        """
        Returns the path to the thumbnail
        """
        info = self.get_dataset_info(dataset_name)
        return info["thumbnail_path"]

    def clear_downloads(self):
        """
        Deletes all downloaded datasets
        """
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

    def _show_citation_notice(self, dataset_name: str):
        info = self.get_dataset_info(dataset_name)
        if info["reference"] not in [None, ""]:
            print(
                f"If you find the '{dataset_name}' dataset useful, please cite: "
                + f"{info['reference']} - {info['reference_doi']}"
            )
