import math
import numpy as np


def calculate_ppmcc(img_ref, img_slice, shift_x, shift_y):
    w = img_ref.shape[1]
    h = img_ref.shape[0]

    new_w = w - abs(shift_x)
    new_h = h - abs(shift_y)

    x0 = max(0, -shift_x)
    y0 = max(0, -shift_y)
    x1 = x0 + shift_x
    y1 = y0 + shift_y

    pixels_1 = img_ref[y0:y0+new_h, x0:x0+new_w].reshape((new_h*new_w))
    pixels_2 = img_slice[y1:y1+new_h, x1:x1+new_w].reshape((new_h*new_w))

    mean_1 = np.mean(pixels_1)
    mean_2 = np.mean(pixels_2)

    covariance = 0.0
    square_sum_1 = 0.0
    square_sum_2 = 0.0

    for i in range(pixels_1.shape[0]):
        v1 = float(pixels_1[i] - mean_1)
        v2 = float(pixels_2[i] - mean_2)

        covariance += (v1 * v2)
        square_sum_1 += (v1 * v1)
        square_sum_2 += (v2 * v2)

    if square_sum_1 == 0 or square_sum_2 == 0:
        return 0
    else:
        return covariance / math.sqrt(square_sum_1 * square_sum_2)

