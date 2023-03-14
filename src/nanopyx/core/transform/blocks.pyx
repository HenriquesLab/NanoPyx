# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

import numpy as np
cimport numpy as np

def assemble_frame_from_blocks(blocks_stack: np.ndarray, n_rows: int, n_cols: int):
    return _assemble_frame_from_blocks(blocks_stack.astype(np.float32), n_rows, n_cols)

cdef float[:, :] _assemble_frame_from_blocks(float[:, :, :] blocks_stack, int n_rows, int n_cols) nogil:

    cdef int block_cols_len = blocks_stack.shape[2]
    cdef int block_rows_len = blocks_stack.shape[1]
    cdef int cols_len = block_cols_len * n_cols
    cdef int rows_len = block_rows_len * n_rows
    cdef int row_i, col_i, count
    cdef float[:, :] reconstructed_image

    with gil:
        assert blocks_stack.shape[0] == n_rows * n_cols
        reconstructed_image = np.empty((rows_len, cols_len), dtype=np.float32)
        assert reconstructed_image.shape[0] == rows_len
        assert reconstructed_image.shape[1] == cols_len
        count = 0
        for row_i in range(n_rows):
            for col_i in range(n_cols):
                reconstructed_image[row_i*block_rows_len:(row_i+1)*block_rows_len, col_i*block_cols_len:(col_i+1)*block_cols_len] = blocks_stack[count]
                count += 1

    return reconstructed_image

def split_frame_into_blocks(img: np.ndarray, n_rows: int, n_cols: int):
    return _split_frame_into_blocks(img, n_rows, n_cols)

cdef float[:, :, :] _split_frame_into_blocks(float[:, :] img, int n_rows, int n_cols) nogil:
    
        cdef int block_cols_len = img.shape[1] // n_cols
        cdef int block_rows_len = img.shape[0] // n_rows
        cdef int cols_len = block_cols_len * n_cols
        cdef int rows_len = block_rows_len * n_rows
        cdef int row_i, col_i, count
        cdef float[:, :, :] blocks_stack
    
        with gil:
            assert img.shape[0] == rows_len
            assert img.shape[1] == cols_len
            blocks_stack = np.empty((n_rows * n_cols, block_rows_len, block_cols_len), dtype=np.float32)
            assert blocks_stack.shape[0] == n_rows * n_cols
            assert blocks_stack.shape[1] == block_rows_len
            assert blocks_stack.shape[2] == block_cols_len
            count = 0
            for row_i in range(n_rows):
                for col_i in range(n_cols):
                    blocks_stack[count] = img[row_i*block_rows_len:(row_i+1)*block_rows_len, col_i*block_cols_len:(col_i+1)*block_cols_len]
                    count += 1
    
        return blocks_stack