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
        self.zip_file_path = zip_file_path
        self.zip_file = zipfile.ZipFile(zip_file_path)
        self.tiff_file_names = [
            name
            for name in self.zip_file.namelist()
            if name.endswith(".tif") and not name.startswith("_")
        ]
        self.tiff_file_names = sorted(self.tiff_file_names)
        # print(self.tiff_file_names)

    def __getitem__(self, index: int) -> np.ndarray:
        if index >= len(self.tiff_file_names):
            raise IndexError
        else:
            with self.zip_file.open(self.tiff_file_names[index]) as tiff_file:
                image = skimage.io.imread(tiff_file)
                return np.array(image)

    def __len__(self) -> int:
        return len(self.tiff_file_names)

    def __iter__(self):
        for index in range(len(self.tiff_file_names)):
            yield self[index]

    def _getIm0(self):
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
        self._getIm0()
        return self._shape

    def get_dtype(self) -> str:
        self._getIm0()
        return self._dtype

    def get_thumb(self, save_path=None):
        thumb = transform.resize(self._getIm0(), (64, 64))
        _max = thumb.max()
        _min = thumb.min()
        thumb = (thumb.astype("float32") - _min / (_max - _min)) * 255
        if save_path != None:
            # Save the thumbnail as a JPEG file
            skimage.io.imsave(os.path.join(save_path, "thumbnail.jpg"), thumb)

    def close(self):
        self.zip_file.close()
