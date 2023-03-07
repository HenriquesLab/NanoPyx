# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

import numpy as np
cimport numpy as np

from cython.parallel import prange

def assemble_frame_from_blocks(blocks_stack: np.ndarray, n_rows: int, n_cols: int):
    return np.array(_assemble_frame_from_blocks(blocks_stack.astype(np.float64), n_rows, n_cols), dtype=np.float32)

cdef double[:, :] _assemble_frame_from_blocks(double[:, :, :] blocks_stack, int n_rows, int n_cols) nogil:

    cdef int block_cols_len = blocks_stack.shape[2]
    cdef int block_rows_len = blocks_stack.shape[1]
    cdef int cols_len = block_cols_len * n_cols
    cdef int rows_len = block_rows_len * n_rows
    cdef double[:, :] reconstructed_image
    cdef int row_i, col_i

    with gil:
        assert blocks_stack.shape[0] == n_rows * n_cols
        reconstructed_image = np.empty((rows_len, cols_len), dtype=np.float64)
        assert reconstructed_image.shape[0] == rows_len
        assert reconstructed_image.shape[1] == cols_len

    for row_i in prange(n_rows):
        for col_i in range(n_cols):
            reconstructed_image[row_i * block_rows_len:(row_i + 1) * block_rows_len,
                                col_i * block_cols_len:(col_i + 1) * block_cols_len] = \
                                blocks_stack[col_i * n_rows + row_i]

    return reconstructed_image