import numpy as np
from math import sqrt
from skimage.filters import gaussian
from skimage.transform import EuclideanTransform, warp

from ..transform.blocks import assemble_frame_from_blocks


def generate_random_position(n_rows, n_cols):
    """
    Generates a random position given number of rows and columns.
    :param n_rows: int; number of rows
    :param n_cols: int; number of columns
    :return: int, int; random position constrained to 0.1 and 0.9 of n_rows and n_cols
    """

    min_r = int(n_rows * 0.1)
    max_r = int(n_rows * 0.9)

    min_c = int(n_cols * 0.1)
    max_c = int(n_cols * 0.9)

    r = np.random.randint(min_r, max_r)
    c = np.random.randint(min_c, max_c)

    return r, c


def generate_image(n_objects=10, shape=(2, 300, 300), dtype=np.float16):
    """
    Generates a random image with objects in random positions
    :param n_objects: int; number of objects to generate
    :param shape: tuple; with shape (z, y, x)
    :param dtype: data type to be used in the generated numpy array
    :return: numpy array with shape (z, y, x) and defined data type and n_objects
    """

    img = np.zeros(shape).astype(dtype)

    n_rows = img.shape[1]
    n_cols = img.shape[2]

    for i in range(n_objects):
        r, c = generate_random_position(n_rows, n_cols)
        img[:, r, c] = np.finfo(np.float16).max

    for i in range(img.shape[0]):
        img[i] = gaussian(img[i], sigma=3)

    return img


def generate_timelapse_drift(
    n_objects=10, shape=(10, 300, 300), dtype=np.float16, drift=None, drift_mode="directional"
):
    """
    Generate random timelapse image with drift over time.
    :param n_objects: int; number of objects to generate
    :param shape: tuple; with shape (t, y, x)
    :param dtype: data type to be used in the generated numpy array
    :param drift: None or int; number of pixels corresponding to drift between frames. If None, automatic drift is
    calculated based on 0.02 of image dimensions.
    :param drift_mode: str; "directional" (default) or "random";
    :return: numpy array with shape (t, y, x) and defined data type and n_objects
    """

    if drift is None:
        drift = min(shape[1] * 0.02, shape[2] * 0.02)

    img = generate_image(n_objects=n_objects, shape=shape, dtype=dtype)

    if drift_mode == "directional":
        transformation_matrix = EuclideanTransform(translation=(-drift, -drift))
        for i in range(shape[0] - 1):
            img[i + 1] = warp(img[i], transformation_matrix.inverse, order=3, preserve_range=True)

    elif drift_mode == "random":
        for i in range(shape[0] - 1):
            state = np.random.randint(0, 3)

            if state == 1:
                transformation_matrix = EuclideanTransform(translation=(-sqrt(drift), -sqrt(drift)))
            elif state == 2:
                transformation_matrix = EuclideanTransform(translation=(-sqrt(drift), sqrt(drift)))
            elif state == 3:
                transformation_matrix = EuclideanTransform(translation=(sqrt(drift), -sqrt(drift)))
            else:
                transformation_matrix = EuclideanTransform(translation=(sqrt(drift), sqrt(drift)))

            img[i + 1] = warp(img[i], transformation_matrix.inverse, order=3, preserve_range=True)

    return img.astype(np.float32)


def generate_channel_misalignment():
    """
    Generates an image with shape (3, 300, 300) with 1 object centered on each 3x3 block of the image.
    Slices corresponding to channel 2 and 3 are shifted relative to channel 1 (template).
    :return: numpy array of shape (3, 300, 300) corresponding to a random image with misalignment between channels.
    """

    n_blocks = 3
    h = 300
    w = 300

    block_img = np.zeros((int(h / n_blocks), int(w / n_blocks)))
    block_h = int(h / n_blocks)
    block_w = int(w / n_blocks)
    block_img[int(block_h / 2), int(block_w / 2)] = 1
    block_img = gaussian(block_img, sigma=3)

    ref_channel = np.zeros((h, w))
    misaligned_blocks = []
    misaligned_blocks_2 = []

    for x_i in range(n_blocks):
        for y_i in range(n_blocks):
            ref_channel[y_i * block_h : y_i * block_h + block_h, x_i * block_w : x_i * block_w + block_w] += block_img

    misalignments = [(-3, -3), (-3, 0), (-3, 3), (0, -3), (0, 0), (0, 3), (3, -3), (3, 0), (3, 3)]

    for mis in misalignments:
        block_img = np.zeros((int(h / n_blocks), int(w / n_blocks)))
        block_h = h / n_blocks
        block_w = w / n_blocks
        block_img[int(block_h / 2) + mis[0], int(block_w / 2) + mis[1]] = 1
        block_img = gaussian(block_img, sigma=3)
        misaligned_blocks.append(block_img)

    misalignments.reverse()

    for mis in misalignments:
        block_img = np.zeros((int(h / n_blocks), int(w / n_blocks)))
        block_h = h / n_blocks
        block_w = w / n_blocks
        block_img[int(block_h / 2) + mis[0], int(block_w / 2) + mis[1]] = 1
        block_img = gaussian(block_img, sigma=3)
        misaligned_blocks_2.append(block_img)

    misaligned_channel = assemble_frame_from_blocks(np.array(misaligned_blocks), 3, 3)
    misaligned_channel_2 = assemble_frame_from_blocks(np.array(misaligned_blocks_2), 3, 3)

    return np.array([ref_channel, misaligned_channel, misaligned_channel_2]).astype(np.float32)
