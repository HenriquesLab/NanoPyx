import numpy as np
from transonic import jit


@jit(backend="numba")
def convolution2D(image, kernel):
    nRows = image.shape[0]
    nCols = image.shape[1]

    nRows_kernel = kernel.shape[0]
    nCols_kernel = kernel.shape[1]

    center_r = (nRows_kernel-1) // 2
    center_c = (nCols_kernel-1) // 2

    acc = 0.0

    conv_out = np.zeros((nRows, nCols), dtype=np.float32)

    for r in range(nRows):
        for c in range(nCols):
            acc = 0
            for kr in range(nRows_kernel):
                for kc in range(nCols_kernel):
                    local_row = min(max(r+(kr-center_r), 0), nRows-1)
                    local_col = min(max(c+(kc-center_c), 0), nCols-1)
                    acc = acc + kernel[kr, kc] * image[local_row, local_col]
            conv_out[r, c] = acc

    return conv_out
