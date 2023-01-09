import numpy as np
from math import sqrt
from scipy.interpolate import interp2d
from skimage.filters import gaussian
from skimage.transform import EuclideanTransform, warp

from ..time.timeit import timeit


def generate_random_position(n_rows, n_cols):

    min_r = int(n_rows * 0.1)
    max_r = int(n_rows * 0.9)

    min_c = int(n_cols * 0.1)
    max_c = int(n_cols * 0.9)

    r = np.random.randint(min_r, max_r)
    c = np.random.randint(min_c, max_c)

    return r, c

def generate_image(n_objects=10, shape=(2, 300, 300), dtype=np.float16):

    img = np.zeros(shape).astype(dtype)

    n_rows = img.shape[1]
    n_cols = img.shape[2]

    for i in range(n_objects):
        r, c = generate_random_position(n_rows, n_cols)
        img[:, r, c] = np.finfo(np.float16).max

    for i in range(img.shape[0]):
        img[i] = gaussian(img[i], sigma=3)

    return img

@timeit
def generate_timelapse_drift(n_objects=10, shape=(10, 300, 300), dtype=np.float16, drift=None,
                             drift_mode="directional"):

    if drift is None:
        drift = min(shape[1]*0.02, shape[2]*0.02)

    img = generate_image(n_objects=n_objects, shape=shape, dtype=dtype)

    if drift_mode == "directional":
        transformation_matrix = EuclideanTransform(translation=(-drift, -drift))
        for i in range(shape[0]-1):
            img[i+1] = warp(img[i], transformation_matrix.inverse, order=3, preserve_range=True)

    elif drift_mode == "random":
        for i in range(shape[0]-1):

            state = np.random.randint(0, 3)

            if state == 1:
                transformation_matrix = EuclideanTransform(translation=(-sqrt(drift), -sqrt(drift)))
            elif state == 2:
                transformation_matrix = EuclideanTransform(translation=(-sqrt(drift), sqrt(drift)))
            elif state == 3:
                transformation_matrix = EuclideanTransform(translation=(sqrt(drift), -sqrt(drift)))
            else:
                transformation_matrix = EuclideanTransform(translation=(sqrt(drift), sqrt(drift)))

            img[i+1] = warp(img[i], transformation_matrix.inverse, order=3, preserve_range=True)

    return img

@timeit
def generate_channel_misalignment(n_objects=10, shape=(2, 300, 300), dtype=np.float16, misalignment=None,
                                  n_blocks=3, misalignment_mode='heterogeneous'):

    if misalignment is None:
        misalignment = min(shape[1]*0.05, shape[2]*0.05)

    r_misalignment = np.linspace(-misalignment, misalignment, n_blocks)
    c_misalignment = np.linspace(-misalignment, misalignment, n_blocks)

    img = generate_image(n_objects=n_objects, shape=shape, dtype=dtype)
    slices, height, width = img.shape

    if misalignment_mode == "constant":
        transformation_matrix = EuclideanTransform(translation=(-misalignment, -misalignment))
        img[1] = warp(img[1], transformation_matrix.inverse, order=3)
    elif misalignment_mode == "heterogeneous":
        block_width = int(img.shape[2]/n_blocks)
        block_height = int(img.shape[1]/n_blocks)

        flow_arrows = []

        for x_i in range(n_blocks):
            for y_i in range(n_blocks):
                x_start = x_i * block_width
                y_start = y_i * block_height

                vector_x = c_misalignment[y_i]
                vector_y = r_misalignment[x_i]
                flow_arrows.append([x_start + block_width/2.0, y_start + block_height/2.0, vector_x, vector_y])

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

        if n_blocks > 1:
            translation_matrix_x = gaussian(translation_matrix_x, sigma=max(block_width, block_height/2.0))
            translation_matrix_y = gaussian(translation_matrix_y, sigma=max(block_width, block_height/2.0))

        translation_matrix[:, :width] += translation_matrix_x
        translation_matrix[:, width:] += translation_matrix_y

        x = [xi for xi in range(img[0].shape[1])]
        y = [yi for yi in range(img[0].shape[0])]
        z = img[0].reshape((img[0].shape[0] * img[0].shape[1]))
        interpolator = interp2d(y, x, z, kind="quintic")
        for y_i in range(height):
            for x_i in range(width):
                dx = translation_matrix[y_i, x_i]
                dy = translation_matrix[y_i, x_i + width]
                value = interpolator(y_i-dy, x_i-dx)
                img[1][y_i, x_i] = value

    return img
