import numpy as np
from cmath import sqrt
from skimage.filters import gaussian

from ..drift.estimate_shift import GetMaxOptimizer
from ..feature_extraction.break_into_blocks import assemble_frame_from_blocks
from ..transform.cross_correlation_map import CrossCorrelationMap

def calculate_translation_mask(img_slice, img_ref, max_shift, blocks_per_axis, min_similarity):

    method = "subpixel" # set for pixel or subpixel precision
    
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
            ccm = CrossCorrelationMap()
            slice_ccm = ccm.calculate_ccm(ref_crop, np.array([slice_crop]), True)[0]

            if max_shift > 0 and max_shift*2+1 < slice_ccm.shape[0] and max_shift*2+1 < slice_ccm.shape[1]:
                ccm_x_start = int(slice_ccm.shape[0]/2 - max_shift)
                ccm_y_start = int(slice_ccm.shape[1]/2 - max_shift)
                slice_ccm = slice_ccm[ccm_y_start:ccm_y_start+max_shift*2+1, ccm_x_start:ccm_x_start+max_shift*2+1]

            if method == "subpixel":
                optimizer = GetMaxOptimizer(slice_ccm)
                max_coords = optimizer.get_max()
                ccm_max_value = -optimizer.get_interpolated_px_value_interp2d(max_coords)
            else:
                max_coords = np.unravel_index(slice_ccm.argmax(), slice_ccm.shape)
                ccm_max_value = slice_ccm[max_coords[0], max_coords[1]]

            ccm_width = slice_ccm.shape[0]
            ccm_height = slice_ccm.shape[1]
            blocks_stack.append(slice_ccm)

            if ccm_max_value >= min_similarity:
                vector_x = ccm_width/2.0 - max_coords[1] - 0.5
                vector_y = ccm_height/2.0 - max_coords[0] - 0.5
                flow_arrows.append([x_start + block_width/2.0, y_start + block_height/2.0, vector_x, vector_y, ccm_max_value])
    
    if len(flow_arrows) == 0:
        print("Couldn't find any correlation between frames... try reducing the 'Min Similarity' parameter")
        return None

    translation_matrix = np.zeros((height, width*2))
    translation_matrix_x = np.zeros((height, width))
    translation_matrix_y = np.zeros((height, width))

    max_distance = sqrt(pow(width, 2) + pow(height, 2))

    for x_i in range(width):
        for y_i in range(height):
            # iterate over vectors
            dx, dy, w_sum = 0, 0, 0

            if len(flow_arrows) == 1:
                dx = flow_arrows[0][2]
                dy = flow_arrows[0][3]

            else:
                distances = []
                for arrow in flow_arrows:
                    distance = sqrt(pow(arrow[0] - x_i, 2) + pow(arrow[1] - y_i, 2))
                    distances.append(distance)
                
                all_distances = 0
                for distance in distances:
                    all_distances += pow(((max_distance - distance) / (max_distance * distance)), 2)

                for i, arrow in enumerate(flow_arrows):
                    distance = distances[i]
                    first_term = pow(((max_distance - distance) / (max_distance * distance)), 2)
                    second_term = all_distances

                    weight = first_term / second_term
                    dx += arrow[2] * weight
                    dy += arrow[3] * weight
                    w_sum += weight

                dx /= w_sum
                dy /= w_sum

            translation_matrix_x[y_i, x_i] = dx
            translation_matrix_y[y_i, x_i] = dy

    if blocks_per_axis > 1:
        translation_matrix_x = gaussian(translation_matrix_x, sigma=max(block_width, block_height/2.0))
        translation_matrix_y = gaussian(translation_matrix_y, sigma=max(block_width, block_height/2.0))

    translation_matrix[:, :width] += translation_matrix_x
    translation_matrix[:, width:] += translation_matrix_y

    blocks = assemble_frame_from_blocks(np.array(blocks_stack), blocks_per_axis, blocks_per_axis)

    return translation_matrix, blocks