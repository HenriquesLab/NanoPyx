
import re
import zipfile

import numpy as np
import skimage.io


def open_tiffs_in_zip(file_path: str, pattern: str = None) -> np.ndarray:
    # Open the zip file
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        # Get a list of the file names in the zip file
        filenames = zip_ref.namelist()

        print(pattern)

        if pattern is not None:
            image_filenames = [
                filename for filename in filenames if re.search(pattern, filename)
            ]
        else:
            image_filenames = [
                filename for filename in filenames if filename.endswith(".tif")
            ]
        image_filenames = [
            filename for filename in image_filenames if not filename.startswith("_")
        ]

        image_filenames = sorted(image_filenames)
        print(image_filenames)

        images = []
        # Find the TIFF file in the list of filenames
        for filename in image_filenames:
            # Open the TIFF file from the zip file
            with zip_ref.open(filename) as tiff_file:
                # Load the TIFF file into a NumPy array using skimage
                image_array = skimage.io.imread(tiff_file)
                images.append(image_array)
                # break

    return np.asarray(images)


# def zip2zarr(file_path: str, pattern: str = None):
#     # Open the zip file
#     with zipfile.ZipFile(file_path, "r") as zip_ref:
#         # Get a list of the file names in the zip file
#         filenames = zip_ref.namelist()

#         if pattern is not None:
#             image_filenames = [
#                 filename for filename in filenames if re.search(pattern, filename)
#             ]
#         else:
#             image_filenames = [
#                 filename for filename in filenames if filename.endswith(".tif")
#             ]
#         image_filenames = [filename for filename in image_filenames if not filename.startswith("_")]
#         image_filenames = sorted(image_filenames)
#         # print(image_filenames)

#         # Create the 3D dataset in the OME-Zarr format
#         new_file_path = os.path.splitext(file_path)[0]+".zarr"
#         store = zarr.ZipStore(new_file_path, mode='w')
#         root = zarr.group(store=store, overwrite=True)
#         root.attrs['OME'] = 'https://www.openmicroscopy.org/Schemas/OME/2016-06'
#         root.attrs['image_count'] = len(image_filenames)
#         dataset = None

#         # Find the TIFF file in the list of filenames
#         for i in range(len(image_filenames)):
#             # Open the TIFF file from the zip file
#             with zip_ref.open(image_filenames[i]) as tiff_file:
#                 # Load the TIFF file into a NumPy array using skimage
#                 image_array = skimage.io.imread(tiff_file)

#                 if dataset == None:
#                     # Create the 3D dataset with dimensions t, x, y
#                     shape = (len(image_filenames), image_array.shape[0], image_array.shape[1])
#                     dtype = image_array.dtype
#                     dataset = root.create_dataset('Data', shape=shape, chunks=True, dtype=dtype, compressor=zarr.Blosc(cname='lz4', clevel=5))

#                 dataset[i] = image_array

#         store.close()
#         return new_file_path
