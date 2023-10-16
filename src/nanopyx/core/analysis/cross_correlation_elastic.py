import numpy as np
from math import sqrt
from skimage.filters import gaussian
from scipy.interpolate import bisplrep, bisplev # replacement for interp2d see: https://gist.github.com/ev-br/8544371b40f414b7eaf3fe6217209bff

from .ccm import calculate_ccm_from_ref
from .estimate_shift import GetMaxOptimizer
from ..transform.blocks import assemble_frame_from_blocks


def calculate_translation_mask(
    img_slice, img_ref, max_shift, blocks_per_axis, min_similarity, method="subpixel", algorithm="weight"
):
    """
    Generate the function comment for the given function body in a markdown code block with the correct language syntax.

    Parameters:
        img_slice (numpy.ndarray): The input image slice.
        img_ref (numpy.ndarray): The reference image.
        max_shift (int): The maximum shift allowed for the translation.
        blocks_per_axis (int): The number of blocks per axis.
        min_similarity (float): The minimum similarity required for a block to be considered a match.
        method (str, optional): The method used for subpixel interpolation. Defaults to "subpixel".
        algorithm (str, optional): The algorithm used for calculating the translation mask. Defaults to "weight".

    Returns:
        numpy.ndarray: The translation mask.

    Raises:
        None
    """

    if algorithm == "weight":
        return calculate_translation_mask_vector_weight(
            img_slice, img_ref, max_shift, blocks_per_axis, min_similarity, method=method
        )
    elif algorithm == "field":
        return calculate_translation_mask_vector_field(
            img_slice, img_ref, max_shift, blocks_per_axis, min_similarity, method=method
        )
    else:
        print("Not a valid algorithm option! Select either 'weight' or 'field'")


def calculate_translation_mask_vector_weight(
    img_slice, img_ref, max_shift, blocks_per_axis, min_similarity, method="subpixel"
):
    """
    Calculates the translation mask vector weight for a given image slice and reference image.

    Parameters:
    - img_slice: The image slice to calculate the translation mask vector weight for.
    - img_ref: The reference image.
    - max_shift: The maximum shift allowed for calculating the translation mask vector weight.
    - blocks_per_axis: The number of blocks per axis for dividing the image.
    - min_similarity: The minimum similarity required for considering a block.
    - method: The method to use for calculating the translation mask vector weight. Defaults to "subpixel".

    Returns:
    - translation_matrix: The translation matrix.
    - blocks: The assembled frame from blocks.

    Raises:
    - AssertionError: If the width and height of the image slice and reference image do not match.

    Note:
    - The translation mask vector weight is calculated by dividing the image into blocks and finding the maximum similarity between the image slice and reference image within each block. The translation vector is then calculated based on the maximum similarity and the block position.
    - The translation mask vector weight is used to determine the translation matrix and the assembled frame from blocks.
    - If no correlation is found between frames, a message is printed and None is returned.
    - If the number of blocks per axis is greater than 1, Gaussian smoothing is applied to the translation matrix.
    """

    width = img_slice.shape[1]
    height = img_slice.shape[0]

    assert width == img_ref.shape[1]
    assert height == img_ref.shape[0]

    block_width = int(width / blocks_per_axis)
    block_height = int(height / blocks_per_axis)

    flow_arrows = []
    blocks_stack = []

    for y_i in range(blocks_per_axis):
        for x_i in range(blocks_per_axis):
            x_start = x_i * block_width
            y_start = y_i * block_height

            slice_crop = img_slice[y_start : y_start + block_height, x_start : x_start + block_width]
            ref_crop = img_ref[y_start : y_start + block_height, x_start : x_start + block_width]
            slice_ccm = np.array(
                calculate_ccm_from_ref(
                    np.array([slice_crop]).astype(np.float32), np.array(ref_crop).astype(np.float32)
                )[0]
            )

            if max_shift > 0 and max_shift * 2 + 1 < slice_ccm.shape[0] and max_shift * 2 + 1 < slice_ccm.shape[1]:
                ccm_x_start = int(slice_ccm.shape[1] / 2 - max_shift)
                ccm_y_start = int(slice_ccm.shape[0] / 2 - max_shift)
                slice_ccm = slice_ccm[
                    ccm_y_start : ccm_y_start + (max_shift * 2), ccm_x_start : ccm_x_start + (max_shift * 2)
                ]

            if method == "subpixel":
                optimizer = GetMaxOptimizer(slice_ccm)
                max_coords = optimizer.get_max()
                ccm_max_value = -optimizer.get_interpolated_px_value(max_coords)
            else:
                max_coords = np.unravel_index(slice_ccm.argmax(), slice_ccm.shape)
                ccm_max_value = slice_ccm[max_coords[0], max_coords[1]]

            ccm_width = slice_ccm.shape[1]
            ccm_height = slice_ccm.shape[0]
            blocks_stack.append(slice_ccm)

            if ccm_max_value >= min_similarity:
                vector_x = ccm_width / 2.0 - max_coords[1] - 1
                vector_y = ccm_height / 2.0 - max_coords[0] - 1
                flow_arrows.append([x_start + block_width / 2.0, y_start + block_height / 2.0, vector_x, vector_y])

    if len(flow_arrows) == 0:
        print("Couldn't find any correlation between frames... try reducing the 'Min Similarity' parameter")
        return None

    translation_matrix = np.zeros((height, width * 2))
    translation_matrix_x = np.zeros((height, width))
    translation_matrix_y = np.zeros((height, width))

    max_distance = sqrt(width * width + height * height)

    for j in range(height):
        for i in range(width):
            # iterate over vectors
            dx, dy, w_sum = 0, 0, 0

            if len(flow_arrows) == 1:
                dx = flow_arrows[0][2]
                dy = flow_arrows[0][3]

            else:
                distances = []
                all_distances = 0
                for arrow in flow_arrows:
                    d = sqrt(pow(arrow[0] - i, 2) + pow(arrow[1] - j, 2)) + 1
                    distances.append(d)
                    all_distances += pow(((max_distance - d) / (max_distance * d)), 2)

                for idx, arrow in enumerate(flow_arrows):
                    d = distances[idx]
                    first_term = pow(((max_distance - d) / (max_distance * d)), 2)
                    second_term = all_distances

                    weight = first_term / second_term
                    dx += arrow[2] * weight
                    dy += arrow[3] * weight
                    w_sum += weight

                dx = dx / w_sum
                dy = dy / w_sum

            translation_matrix_x[j, i] = dx
            translation_matrix_y[j, i] = dy

    if blocks_per_axis > 1:
        translation_matrix_x = gaussian(translation_matrix_x, sigma=max(block_width, block_height / 2.0))
        translation_matrix_y = gaussian(translation_matrix_y, sigma=max(block_width, block_height / 2.0))

    translation_matrix[:, :width] += translation_matrix_x
    translation_matrix[:, width:] += translation_matrix_y

    blocks = assemble_frame_from_blocks(np.array(blocks_stack), blocks_per_axis, blocks_per_axis)

    return translation_matrix, blocks


def calculate_translation_mask_vector_field(
    img_slice, img_ref, max_shift, blocks_per_axis, min_similarity, method="subpixel"
):
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

            slice_crop = img_slice[y_start : y_start + block_height, x_start : x_start + block_width]
            ref_crop = img_ref[y_start : y_start + block_height, x_start : x_start + block_width]
            slice_ccm = np.array(
                calculate_ccm_from_ref(
                    np.array([slice_crop]).astype(np.float32), np.array(ref_crop).astype(np.float32)
                )[0]
            )

            ccm_x_start = 0
            ccm_y_start = 0

            if max_shift > 0 and max_shift * 2 + 1 < slice_ccm.shape[0] and max_shift * 2 + 1 < slice_ccm.shape[1]:
                ccm_x_start = int(slice_ccm.shape[1] / 2 - max_shift)
                ccm_y_start = int(slice_ccm.shape[0] / 2 - max_shift)
                slice_ccm = slice_ccm[
                    ccm_y_start : ccm_y_start + (max_shift * 2), ccm_x_start : ccm_x_start + (max_shift * 2)
                ]

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
                y_translation.append(
                    [y_start + max_coords[0] + ccm_y_start, x_start + max_coords[1] + ccm_x_start, shift_y - 0.5]
                )
                x_translation.append(
                    [y_start + max_coords[0] + ccm_y_start, x_start + max_coords[1] + ccm_x_start, shift_x - 0.5]
                )

    y_translation = np.array(y_translation)
    x_translation = np.array(x_translation)

    y_spline = bisplrep(y_translation[:, 0], y_translation[:, 1], y_translation[:, 2],kx=1, ky=1, s=0) # bspline representation
    x_spline = bisplrep(x_translation[:, 0], x_translation[:, 1], x_translation[:, 2],kx=1, ky=1, s=0)
    x_interp = lambda x, y: bisplev(x, y, x_spline).T  # bspline evaluation
    y_interp = lambda x, y: bisplev(x, y, y_spline).T   

    translation_matrix = np.zeros((height, width * 2))
    translation_matrix_x = np.zeros((height, width))
    translation_matrix_y = np.zeros((height, width))

    for j in range(translation_matrix_x.shape[0]):
        for i in range(translation_matrix_x.shape[1]):
            translation_matrix_x[j, i] = x_interp(j, i)
            translation_matrix_y[j, i] = y_interp(j, i)

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
