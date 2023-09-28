import os
import zipfile

import numpy as np
import skimage.io
from skimage import transform


class ZipTiffIterator:
    _shape = None
    _dtype = None
    _im0 = None

    def __init__(self, zip_file_path: str):
        """
        Iterator to sequentially open tiff files contained in a zip file
        :param zip_file_path: path to the zip file
        :type zip_file_path: str
        """
        self.zip_file_path = zip_file_path
        self.zip_file = zipfile.ZipFile(zip_file_path)
        self.tiff_file_names = [
            name for name in self.zip_file.namelist() if name.endswith(".tif") and not name.startswith("_")
        ]
        self.tiff_file_names = sorted(self.tiff_file_names)
        # print(self.tiff_file_names)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Get the item at the specified index.

        Parameters:
            index (int): The index of the item to get.

        Returns:
            np.ndarray: The item at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index >= len(self.tiff_file_names):
            raise IndexError
        else:
            with self.zip_file.open(self.tiff_file_names[index]) as tiff_file:
                image = skimage.io.imread(tiff_file)
                return np.array(image)

    def __len__(self) -> int:
        """
        Returns the length of the `tiff_file_names` list.

        :return: An integer representing the length of the `tiff_file_names` list.
        :rtype: int
        """
        return len(self.tiff_file_names)

    def __iter__(self):
        """
        Returns an iterator object that iterates over the elements of the class.

        Yields:
            The elements of the class.
        """
        for index in range(len(self.tiff_file_names)):
            yield self[index]

    def _get_im0(self):
        """
        Get the first image in the stack.

        :return: The first image in the stack.
        """
        if self._im0 is None:
            self._im0 = self[0]
            self._shape = (
                len(self.tiff_file_names),
                self._im0.shape[0],
                self._im0.shape[1],
            )
            self._dtype = self._im0.dtype
        return self._im0

    def get_shape(self) -> list:
        """
        Returns the shape of the image stack
        :return: shape of the image stack
        :rtype: list
        """
        self._get_im0()
        return self._shape

    def get_dtype(self) -> str:
        """
        Returns the data type of the image stack
        :return: data type of the image stack
        :rtype: str
        """
        self._get_im0()
        return self._dtype

    def get_thumb(self, save_path=None):
        """
        Saves a thumbnail (first frame) of the image stack
        :param save_path: path to save the thumbnail
        :type save_path: str
        """
        thumb = transform.resize(self._get_im0(), (64, 64))
        _max = thumb.max()
        _min = thumb.min()
        thumb = (thumb.astype("float32") - _min / (_max - _min)) * 255
        if save_path != None:
            # Save the thumbnail as a JPEG file
            skimage.io.imsave(os.path.join(save_path, "thumbnail.jpg"), thumb)

    def close(self):
        """
        Closes the zip file
        """
        self.zip_file.close()
