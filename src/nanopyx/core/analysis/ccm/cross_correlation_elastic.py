import numpy as np
from math import sqrt
from skimage.filters import gaussian

from .ccm import calculate_ccm_from_ref
from .estimate_shift import GetMaxOptimizer
from ...image.blocks import assemble_frame_from_blocks
from ...transform.image_magnify import catmull_rom_zoom_xy


def calculate_translation_mask(img_slice, img_ref, max_shift, blocks_per_axis, min_similarity, method="subpixel"):
    """
    Function used to calculate a translation mask between 2 different images.
    Method based on dividing both images in smaller blocks and calculate cross correlation matrix between corresponding
    blocks. From the ccm, the shift between the two images is calculated for each block and a translation matrix is
    using the shifts in the center position of each block and then interpolating the remaining translation mask.
    :param img_slice: numpy array with shape (y, x); image to be used for translation mask calculation
    :param img_ref: numpy array with shape (y, x); image to be used as reference for translation mask calculation
    :param max_shift: int; maximum shift accepted between each corresponding block, in pixels.
    :param blocks_per_axis: int; number of blocks for both axis
    :param min_similarity: float; minimum similarity (cross correlation value after normalization) between corresponding
    blocks.
    :param method: str, either "subpixel" or "max"; defaults to "subpixel"; subpixel uses a minimizer to achieve
    subpixel precision in shift calculation. max simply takes the coordinates corresponding to the max value of the ccm.
    :return: numpy array with shape (y, x) where width is equal to two times the width of the original image.
    [:, :width/2] corresponds to the translation mask for x and [:, width/2:] corresponds to the translation mask for y.
    """

    width = img_slice.shape[1]
    height = img_slice.shape[0]

    assert width == img_ref.shape[1]
    assert height == img_ref.shape[0]

    block_width = int(width / blocks_per_axis)
    block_height = int(height / blocks_per_axis)

    blocks_stack = []

    y_translation = []
    x_translation = []

    for y_i in range(blocks_per_axis):
        for x_i in range(blocks_per_axis):
            x_start = x_i * block_width
            y_start = y_i * block_height

            slice_crop = img_slice[y_start:y_start+block_height, x_start:x_start+block_width]
            ref_crop = img_ref[y_start:y_start+block_height, x_start:x_start+block_width]
            slice_ccm = np.array(calculate_ccm_from_ref(np.array([slice_crop]).astype(np.float32),
                                                        np.array(ref_crop).astype(np.float32))[0])

            ccm_x_start = 0
            ccm_y_start = 0

            if max_shift > 0 and max_shift*2+1 < slice_ccm.shape[0] and max_shift*2+1 < slice_ccm.shape[1]:
                ccm_x_start = int(slice_ccm.shape[1]/2 - max_shift)
                ccm_y_start = int(slice_ccm.shape[0]/2 - max_shift)
                slice_ccm = slice_ccm[ccm_y_start:ccm_y_start+(max_shift*2), ccm_x_start:ccm_x_start+(max_shift*2)]

            if method == "subpixel":
                optimizer = GetMaxOptimizer(slice_ccm)
                max_coords = optimizer.get_max()
                ccm_max_value = -optimizer.get_interpolated_px_value(max_coords)
            else:
                max_coords = np.unravel_index(slice_ccm.argmax(), slice_ccm.shape)
                ccm_max_value = slice_ccm[max_coords[0], max_coords[1]]

            blocks_stack.append(slice_ccm)

            if ccm_max_value >= min_similarity:
                shift_x, shift_y = get_shift_from_ccm_slice(slice_ccm, method=method)
                y_translation.append([shift_y - 0.5])
                x_translation.append([shift_x - 0.5])

    y_translation = np.array(y_translation).astype(np.float32).reshape((blocks_per_axis, blocks_per_axis))
    x_translation = np.array(x_translation).astype(np.float32).reshape((blocks_per_axis, blocks_per_axis))
    print(y_translation, x_translation)

    translation_matrix = np.empty((height, width*2))
    translation_matrix_x = catmull_rom_zoom_xy(x_translation, magnification_y=int(height/blocks_per_axis), magnification_x=int(width/blocks_per_axis))
    translation_matrix_y = catmull_rom_zoom_xy(y_translation, magnification_y=int(height/blocks_per_axis), magnification_x=int(width/blocks_per_axis))
    print(translation_matrix_x.shape, translation_matrix_y.shape)

    translation_matrix[:, :width] += translation_matrix_x
    translation_matrix[:, width:] += translation_matrix_y

    blocks = assemble_frame_from_blocks(np.array(blocks_stack), blocks_per_axis, blocks_per_axis)

    return translation_matrix, blocks

def get_shift_from_ccm_slice(slice_ccm, method="subpixel"):
    """
    Function used to calculate the shift corresponding to the maximum correlation between two images.
    :param slice_ccm: numpy array with shape (y, x);
    :param method: str, either "subpixel" or "max"; defaults to "subpixel"; subpixel uses a minimizer to achieve
    subpixel precision in shift calculation. max simply takes the coordinates corresponding to the max value of the ccm.
    :return: tuple of floats; values corresponding to shift_x and shift_y, in this order.
    """

    w = slice_ccm.shape[1]
    h = slice_ccm.shape[0]

    radius_x = w / 2.0
    radius_y = h / 2.0

    if method == "subpixel":
        optimizer = GetMaxOptimizer(slice_ccm)
        shift_y, shift_x = optimizer.get_max()
    elif method == "Max":
        shift_y, shift_x = np.unravel_index(slice_ccm.argmax(), slice_ccm.shape)

    shift_x = radius_x - shift_x - 0.5
    shift_y = radius_y - shift_y - 0.5

    return (shift_x, shift_y)
