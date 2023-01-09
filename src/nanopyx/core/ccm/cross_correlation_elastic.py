import numpy as np
from cmath import sqrt
from skimage.filters import gaussian

from .estimate_shift import GetMaxOptimizer
from ..image.blocks import assemble_frame_from_blocks
from ..ccm.ccm import calculate_ccm_from_ref


def calculate_translation_mask(img_slice, img_ref, max_shift, blocks_per_axis, min_similarity, method="subpixel"):

    method = method # set for pixel or subpixel precision
    
    width = img_slice.shape[1]
    height = img_slice.shape[0]

    assert width == img_ref.shape[1]
    assert height == img_ref.shape[0]

    block_width = int(width / blocks_per_axis)
    block_height = int(height / blocks_per_axis)

    flow_arrows = []
    blocks_stack = []

    for x_i in range(blocks_per_axis):
        for y_i in range(blocks_per_axis):
            x_start = x_i * block_width
            y_start = y_i * block_height

            slice_crop = img_slice[y_start:y_start+block_height, x_start:x_start+block_width]
            ref_crop = img_ref[y_start:y_start+block_height, x_start:x_start+block_width]
            slice_ccm = np.array(calculate_ccm_from_ref(np.array([slice_crop]).astype(np.float32), np.array(ref_crop).astype(np.float32))[0])

            if max_shift > 0 and max_shift*2+1 < slice_ccm.shape[0] and max_shift*2+1 < slice_ccm.shape[1]:
                ccm_x_start = int(slice_ccm.shape[1]/2 - max_shift)
                ccm_y_start = int(slice_ccm.shape[0]/2 - max_shift)
                slice_ccm = slice_ccm[ccm_y_start:ccm_y_start+max_shift*2+1, ccm_x_start:ccm_x_start+max_shift*2+1]

            if method == "subpixel":
                optimizer = GetMaxOptimizer(slice_ccm)
                max_coords = optimizer.get_max()
                ccm_max_value = -optimizer.get_interpolated_px_value_interp2d(max_coords)
            else:
                max_coords = np.unravel_index(slice_ccm.argmax(), slice_ccm.shape)
                ccm_max_value = slice_ccm[max_coords[0], max_coords[1]]

            ccm_width = slice_ccm.shape[1]
            ccm_height = slice_ccm.shape[0]
            blocks_stack.append(slice_ccm)

            if ccm_max_value >= min_similarity:
                vector_x = (ccm_width/2.0 - max_coords[1] - 0.5)
                vector_y = (ccm_height/2.0 - max_coords[0] - 0.5)
                flow_arrows.append([x_start + block_width/2.0, y_start + block_height/2.0, vector_x, vector_y])
                
    if len(flow_arrows) == 0:
        print("Couldn't find any correlation between frames... try reducing the 'Min Similarity' parameter")
        return None

    translation_matrix = np.zeros((height, width*2))
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
        translation_matrix_x = gaussian(translation_matrix_x, sigma=max(block_width, block_height/2.0))
        translation_matrix_y = gaussian(translation_matrix_y, sigma=max(block_width, block_height/2.0))

    translation_matrix[:, :width] += translation_matrix_x
    translation_matrix[:, width:] += translation_matrix_y

    blocks = assemble_frame_from_blocks(np.array(blocks_stack), blocks_per_axis, blocks_per_axis)

    return translation_matrix, blocks

def calculate_translation_mask_from_ccm(img_slice, ccm, max_shift, blocks_per_axis, min_similarity, method="subpixel"):

    method = method # set for pixel or subpixel precision
    
    width = img_slice.shape[1]
    height = img_slice.shape[0]

    block_width = int(width / blocks_per_axis)
    block_height = int(height / blocks_per_axis)

    ccm_width = int(ccm.shape[1]/blocks_per_axis)
    ccm_height = int(ccm.shape[0]/blocks_per_axis)

    flow_arrows = []
    blocks_stack = []

    for x_i in range(blocks_per_axis):
        for y_i in range(blocks_per_axis):
            x_start = x_i * block_width
            y_start = y_i * block_height

            ccm_x_start = x_i * ccm_width
            ccm_y_start = y_i * ccm_height

            slice_ccm = ccm[ccm_y_start:ccm_y_start+ccm_height, ccm_x_start:ccm_x_start+ccm_width]

            if method == "subpixel":
                optimizer = GetMaxOptimizer(slice_ccm)
                max_coords = optimizer.get_max()
                ccm_max_value = -optimizer.get_interpolated_px_value_interp2d(max_coords)
            else:
                max_coords = np.unravel_index(slice_ccm.argmax(), slice_ccm.shape)
                ccm_max_value = slice_ccm[max_coords[0], max_coords[1]]

            ccm_width = slice_ccm.shape[1]
            ccm_height = slice_ccm.shape[0]
            blocks_stack.append(slice_ccm)

            if ccm_max_value >= min_similarity:
                vector_x = ccm_width/2.0 - max_coords[1] - 0.5
                vector_y = ccm_height/2.0 - max_coords[0] - 0.5
                flow_arrows.append([x_start + block_width/2.0, y_start + block_height/2.0, vector_x, vector_y])
                
    if len(flow_arrows) == 0:
        print("Couldn't find any correlation between frames... try reducing the 'Min Similarity' parameter")
        return None

    translation_matrix = np.zeros((height, width*2))
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
        translation_matrix_x = gaussian(translation_matrix_x, sigma=max(block_width, block_height/2.0))
        translation_matrix_y = gaussian(translation_matrix_y, sigma=max(block_width, block_height/2.0))
    translation_matrix[:, :width] += translation_matrix_x
    translation_matrix[:, width:] += translation_matrix_y

    blocks = assemble_frame_from_blocks(np.array(blocks_stack), blocks_per_axis, blocks_per_axis)

    return translation_matrix, blocks