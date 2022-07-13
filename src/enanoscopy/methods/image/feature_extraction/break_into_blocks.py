import numpy as np


def assemble_frame_from_blocks(blocks_stack, n_blocks_width, n_blocks_height):
    assert blocks_stack.shape[0] == n_blocks_height * n_blocks_width

    width_b = blocks_stack.shape[2]
    height_b = blocks_stack.shape[1]
    width = width_b * n_blocks_width
    height = height_b * n_blocks_height

    reconstructed_image = np.zeros((height, width))

    counter = 0
    for x_i in range(n_blocks_width):
        for y_i in range(n_blocks_height):
            reconstructed_image[y_i*height_b:y_i*height_b+height_b, x_i*width_b:x_i*width_b+width_b] += blocks_stack[counter]
            counter += 1

    return reconstructed_image