# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

import numpy as np
from skimage.filters import gaussian

cimport numpy as np
from cython.parallel import parallel, prange
from libc.math cimport sqrt,pow

from ...__liquid_engine__ import LiquidEngine
from ...__opencl__ import cl, cl_array
from .ccm cimport _calculate_slice_ccm

from ..transform import CRShiftAndMagnify

cdef bint _check_even_square(float[:, :] image_arr) nogil:
    cdef int r = image_arr.shape[0]
    cdef int c = image_arr.shape[1]

    if r != c:
        return False
    if c % 2 != 0:
        return False

    return True

cdef float[:, :] _make_even_square(float[:, :] image_arr) nogil:
    if _check_even_square(image_arr):
        return image_arr

    cdef int c = image_arr.shape[1]
    cdef int r = image_arr.shape[0]
    cdef int min_size = _get_closest_even_square_size(image_arr)
    cdef int r_start, r_finish, c_start, c_finish

    r_start = (r-min_size)//2
    if (r-min_size) % 2 != 0:
        r_finish = r - (r-min_size) // 2 - 1
    else:
        r_finish = r - (r-min_size) // 2

    c_start = int((c-min_size)/2)
    if (c - min_size) % 2 != 0:
        c_finish = c - (c-min_size) // 2 - 1
    else:
        c_finish = c - (c-min_size) // 2 

    return image_arr[r_start:r_finish, c_start:c_finish]

cdef int _get_closest_even_square_size(float[:, :] image_arr) nogil:
    cdef int c = image_arr.shape[1]
    cdef int r = image_arr.shape[0]
    cdef int min_size = min(r, c)

    if min_size % 2 != 0:
        min_size -= 1

    return min_size

cdef float[:, :] _calculate_ccm_from_ref(float[:, :] img_stack, float[:, :] img_ref):

    if not _check_even_square(img_stack):
        img_stack = _make_even_square(img_stack)

    if not _check_even_square(img_ref):
        img_ref = _make_even_square(img_ref)

    cdef int stack_c = img_stack.shape[1]
    cdef int stack_r = img_stack.shape[0]

    cdef float[:,:] ccm = _calculate_slice_ccm(img_ref, img_stack)

    return ccm

class ChannelRegistrationEstimator(LiquidEngine):
    """
    Channel Registration Estimator
    """

    def __init__(self, clear_benchmarks=False, testing=False):
        self._designation = "ChannelRegistrationEstimator"
        super().__init__(clear_benchmarks=clear_benchmarks, testing=testing, 
                         unthreaded_=True, threaded_=True, threaded_static_=True, threaded_dynamic_=True, threaded_guided_=True)
        
    def run(self, img_stack, img_ref, max_shift, blocks_per_axis, min_similarity, run_type=None):
        return self._run(img_stack, img_ref, max_shift, blocks_per_axis, min_similarity, run_type=run_type)

    def benchmark(self, img_stack, img_ref, max_shift, blocks_per_axis, min_similarity):
        return super().benchmark(img_stack, img_ref, max_shift, blocks_per_axis, min_similarity)

    def _run_unthreaded(self, float[:,:,:] img_stack, float[:,:] img_ref, int max_shift, int blocks_per_axis, float min_similarity):

        crsm = CRShiftAndMagnify()
        
        cdef int nChannels =  img_stack.shape[0]    
        cdef int nRows =  img_stack.shape[1]
        cdef int nCols = img_stack.shape[2]    

        cdef int block_nRows = nRows // blocks_per_axis
        cdef int block_nCols = nCols // blocks_per_axis

        cdef float[:,:,:] translation_masks = np.zeros((nChannels, nRows, nCols * 2)).astype(np.float32)
    
        cdef int channel, row_i, col_i
        cdef int row_start, col_start
        cdef int ccm_col_start, ccm_row_start
        cdef float[:,:] img_slice 
        cdef float[:,:] slice_crop, ref_crop, slice_ccm

        cdef float[:,:] flow_arrows = np.zeros((blocks_per_axis**2,4)).astype(np.float32)
        cdef int n_flow_arrows = 0

        cdef int max_coords_r, max_coords_c
        cdef float[:,:] upscaled_ccm_slice
        cdef float vector_c, vector_r, ccm_max_value

        cdef int ccm_cols, ccm_rows
        translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)
        translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
        cdef float[:,:] _translation_matrix_c = translation_matrix_c
        cdef float[:,:] _translation_matrix_r = translation_matrix_r

        cdef int j,i,n_arrow,idx
        cdef float dx,dy,w_sum,max_distance,all_distances
        cdef float[:] distances 
        cdef float d, first_term, second_term, weight

        for channel in range(nChannels):
            img_slice = img_stack[channel,:,:]
            n_flow_arrows = 0
            for row_i in range(blocks_per_axis):
                for col_i in range(blocks_per_axis):
                    row_start = row_i * block_nRows
                    col_start = col_i * block_nCols

                    slice_crop = img_slice[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    ref_crop = img_ref[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    slice_ccm = _calculate_ccm_from_ref(slice_crop, ref_crop)

                    if max_shift > 0 and max_shift * 2 + 1 < slice_ccm.shape[0] and max_shift * 2 + 1 < slice_ccm.shape[1]:
                        ccm_col_start = slice_ccm.shape[1] // 2 - max_shift
                        ccm_row_start = slice_ccm.shape[0] // 2 - max_shift
                        slice_ccm = slice_ccm[ccm_row_start:ccm_row_start+(max_shift*2), ccm_col_start:ccm_col_start + (max_shift * 2)]

                    max_idx = np.unravel_index(np.argmax(slice_ccm), np.array(slice_ccm).shape)
                    # Upscale 10x around max_idx                     
                    upscaled_ccm_slice = crsm.run(np.ascontiguousarray(slice_ccm),0,0,10,10)[0]                    
                    max_idx_upscaled = np.unravel_index(np.argmax(upscaled_ccm_slice), np.array(upscaled_ccm_slice).shape)
                    max_coords_r = (max_idx_upscaled[0]//10)
                    max_coords_c = (max_idx_upscaled[1]//10)
                    ccm_max_value = upscaled_ccm_slice[max_idx_upscaled[0],max_idx_upscaled[1]]

                    print(max_idx,max_coords_r,max_coords_c)
                    print(slice_ccm[max_idx[0],max_idx[1]], ccm_max_value)

                    ccm_cols = slice_ccm.shape[1]
                    ccm_rows = slice_ccm.shape[0]
                    if ccm_max_value >= min_similarity:
                        vector_c = ccm_cols / 2.0 - max_coords_c - 1 
                        vector_r = ccm_rows / 2.0 - max_coords_r - 1
                        flow_arrows[n_flow_arrows,0] = col_start+block_nCols/2.0
                        flow_arrows[n_flow_arrows,1] = row_start+block_nRows/2.0
                        flow_arrows[n_flow_arrows,2] = vector_c
                        flow_arrows[n_flow_arrows,3] = vector_r
                        n_flow_arrows += 1

            if n_flow_arrows==0:
                return None

            translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
            translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)

            max_distance = sqrt(nRows * nRows + nCols * nCols)

            for j in range(nRows):
                for i in range(nCols):
                    # iterate over vectors
                    dx = 0
                    dy = 0 
                    w_sum = 0

                    if n_flow_arrows == 1:
                        dx = flow_arrows[0,2]
                        dy = flow_arrows[0,3]
                    else:

                        distances = np.zeros(n_flow_arrows).astype(np.float32)
                        all_distances = 0
                        for n_arrow in range(n_flow_arrows):
                            d = sqrt(pow(flow_arrows[n_arrow,0] - i, 2) + pow(flow_arrows[n_arrow,1] - j, 2)) + 1
                            distances[n_arrow] = d
                            all_distances += pow(((max_distance - d) / (max_distance * d)), 2)

                        for idx in range(n_flow_arrows):
                            d = distances[idx]
                            first_term = pow(((max_distance - d) / (max_distance * d)), 2)
                            second_term = all_distances

                            weight = first_term / second_term
                            dx += flow_arrows[idx,2] * weight
                            dy += flow_arrows[idx,3] * weight
                            w_sum += weight

                        dx = dx / w_sum
                        dy = dy / w_sum

                    _translation_matrix_c[j, i] = dx
                    _translation_matrix_r[j, i] = dy

            if blocks_per_axis > 1:
                translation_matrix_c = gaussian(translation_matrix_c, sigma=max(block_nCols, block_nRows / 2.0))
                translation_matrix_r = gaussian(translation_matrix_r, sigma=max(block_nCols, block_nRows / 2.0))

            translation_masks[channel,:,:nCols] = _translation_matrix_c
            translation_masks[channel,:, nCols:] = _translation_matrix_r

        return np.array(translation_masks)

    def _run_threaded(self, float[:,:,:] img_stack, float[:,:] img_ref, int max_shift, int blocks_per_axis, float min_similarity):

        crsm = CRShiftAndMagnify()
        
        cdef int nChannels =  img_stack.shape[0]    
        cdef int nRows =  img_stack.shape[1]
        cdef int nCols = img_stack.shape[2]    

        cdef int block_nRows = nRows // blocks_per_axis
        cdef int block_nCols = nCols // blocks_per_axis

        cdef float[:,:,:] translation_masks = np.zeros((nChannels, nRows, nCols * 2)).astype(np.float32)
    
        cdef int channel, row_i, col_i
        cdef int row_start, col_start
        cdef int ccm_col_start, ccm_row_start
        cdef float[:,:] img_slice 
        cdef float[:,:] slice_crop, ref_crop, slice_ccm

        cdef float[:,:] flow_arrows = np.zeros((blocks_per_axis**2,4)).astype(np.float32)
        cdef int n_flow_arrows = 0

        cdef int max_coords_r, max_coords_c
        cdef float[:,:] upscaled_ccm_slice
        cdef float vector_c, vector_r, ccm_max_value

        cdef int ccm_cols, ccm_rows
        translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)
        translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
        cdef float[:,:] _translation_matrix_c = translation_matrix_c
        cdef float[:,:] _translation_matrix_r = translation_matrix_r

        cdef int j,i,n_arrow,idx
        cdef float dx,dy,w_sum,max_distance,all_distances
        cdef float[:] distances 
        cdef float d, first_term, second_term, weight

        for channel in range(nChannels):
            img_slice = img_stack[channel,:,:]
            n_flow_arrows = 0
            for row_i in range(blocks_per_axis):
                for col_i in range(blocks_per_axis):
                    row_start = row_i * block_nRows
                    col_start = col_i * block_nCols

                    slice_crop = img_slice[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    ref_crop = img_ref[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    slice_ccm = _calculate_ccm_from_ref(slice_crop, ref_crop)

                    if max_shift > 0 and max_shift * 2 + 1 < slice_ccm.shape[0] and max_shift * 2 + 1 < slice_ccm.shape[1]:
                        ccm_col_start = slice_ccm.shape[1] // 2 - max_shift
                        ccm_row_start = slice_ccm.shape[0] // 2 - max_shift
                        slice_ccm = slice_ccm[ccm_row_start:ccm_row_start+(max_shift*2), ccm_col_start:ccm_col_start + (max_shift * 2)]

                    max_idx = np.unravel_index(np.argmax(slice_ccm), np.array(slice_ccm).shape)
                    # Upscale 10x around max_idx                     
                    upscaled_ccm_slice = crsm.run(np.ascontiguousarray(slice_ccm),0,0,10,10)[0]                    
                    max_idx_upscaled = np.unravel_index(np.argmax(upscaled_ccm_slice), np.array(upscaled_ccm_slice).shape)
                    max_coords_r = (max_idx_upscaled[0]//10)
                    max_coords_c = (max_idx_upscaled[1]//10)
                    ccm_max_value = upscaled_ccm_slice[max_idx_upscaled[0],max_idx_upscaled[1]]

                    print(max_idx,max_coords_r,max_coords_c)
                    print(slice_ccm[max_idx[0],max_idx[1]], ccm_max_value)

                    ccm_cols = slice_ccm.shape[1]
                    ccm_rows = slice_ccm.shape[0]
                    if ccm_max_value >= min_similarity:
                        vector_c = ccm_cols / 2.0 - max_coords_c - 1 
                        vector_r = ccm_rows / 2.0 - max_coords_r - 1
                        flow_arrows[n_flow_arrows,0] = col_start+block_nCols/2.0
                        flow_arrows[n_flow_arrows,1] = row_start+block_nRows/2.0
                        flow_arrows[n_flow_arrows,2] = vector_c
                        flow_arrows[n_flow_arrows,3] = vector_r
                        n_flow_arrows += 1

            if n_flow_arrows==0:
                return None

            translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
            translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)

            max_distance = sqrt(nRows * nRows + nCols * nCols)

            for j in range(nRows):
                for i in range(nCols):
                    # iterate over vectors
                    dx = 0
                    dy = 0 
                    w_sum = 0

                    if n_flow_arrows == 1:
                        dx = flow_arrows[0,2]
                        dy = flow_arrows[0,3]
                    else:

                        distances = np.zeros(n_flow_arrows).astype(np.float32)
                        all_distances = 0
                        for n_arrow in range(n_flow_arrows):
                            d = sqrt(pow(flow_arrows[n_arrow,0] - i, 2) + pow(flow_arrows[n_arrow,1] - j, 2)) + 1
                            distances[n_arrow] = d
                            all_distances += pow(((max_distance - d) / (max_distance * d)), 2)

                        for idx in range(n_flow_arrows):
                            d = distances[idx]
                            first_term = pow(((max_distance - d) / (max_distance * d)), 2)
                            second_term = all_distances

                            weight = first_term / second_term
                            dx += flow_arrows[idx,2] * weight
                            dy += flow_arrows[idx,3] * weight
                            w_sum += weight

                        dx = dx / w_sum
                        dy = dy / w_sum

                    _translation_matrix_c[j, i] = dx
                    _translation_matrix_r[j, i] = dy

            if blocks_per_axis > 1:
                translation_matrix_c = gaussian(translation_matrix_c, sigma=max(block_nCols, block_nRows / 2.0))
                translation_matrix_r = gaussian(translation_matrix_r, sigma=max(block_nCols, block_nRows / 2.0))

            translation_masks[channel,:,:nCols] = _translation_matrix_c
            translation_masks[channel,:, nCols:] = _translation_matrix_r

        return np.array(translation_masks)

    def _run_threaded_guided(self, float[:,:,:] img_stack, float[:,:] img_ref, int max_shift, int blocks_per_axis, float min_similarity):

        crsm = CRShiftAndMagnify()
        
        cdef int nChannels =  img_stack.shape[0]    
        cdef int nRows =  img_stack.shape[1]
        cdef int nCols = img_stack.shape[2]    

        cdef int block_nRows = nRows // blocks_per_axis
        cdef int block_nCols = nCols // blocks_per_axis

        cdef float[:,:,:] translation_masks = np.zeros((nChannels, nRows, nCols * 2)).astype(np.float32)
    
        cdef int channel, row_i, col_i
        cdef int row_start, col_start
        cdef int ccm_col_start, ccm_row_start
        cdef float[:,:] img_slice 
        cdef float[:,:] slice_crop, ref_crop, slice_ccm

        cdef float[:,:] flow_arrows = np.zeros((blocks_per_axis**2,4)).astype(np.float32)
        cdef int n_flow_arrows = 0

        cdef int max_coords_r, max_coords_c
        cdef float[:,:] upscaled_ccm_slice
        cdef float vector_c, vector_r, ccm_max_value

        cdef int ccm_cols, ccm_rows
        translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)
        translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
        cdef float[:,:] _translation_matrix_c = translation_matrix_c
        cdef float[:,:] _translation_matrix_r = translation_matrix_r

        cdef int j,i,n_arrow,idx
        cdef float dx,dy,w_sum,max_distance,all_distances
        cdef float[:] distances 
        cdef float d, first_term, second_term, weight

        for channel in range(nChannels):
            img_slice = img_stack[channel,:,:]
            n_flow_arrows = 0
            for row_i in range(blocks_per_axis):
                for col_i in range(blocks_per_axis):
                    row_start = row_i * block_nRows
                    col_start = col_i * block_nCols

                    slice_crop = img_slice[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    ref_crop = img_ref[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    slice_ccm = _calculate_ccm_from_ref(slice_crop, ref_crop)

                    if max_shift > 0 and max_shift * 2 + 1 < slice_ccm.shape[0] and max_shift * 2 + 1 < slice_ccm.shape[1]:
                        ccm_col_start = slice_ccm.shape[1] // 2 - max_shift
                        ccm_row_start = slice_ccm.shape[0] // 2 - max_shift
                        slice_ccm = slice_ccm[ccm_row_start:ccm_row_start+(max_shift*2), ccm_col_start:ccm_col_start + (max_shift * 2)]

                    max_idx = np.unravel_index(np.argmax(slice_ccm), np.array(slice_ccm).shape)
                    # Upscale 10x around max_idx                     
                    upscaled_ccm_slice = crsm.run(np.ascontiguousarray(slice_ccm),0,0,10,10)[0]                    
                    max_idx_upscaled = np.unravel_index(np.argmax(upscaled_ccm_slice), np.array(upscaled_ccm_slice).shape)
                    max_coords_r = (max_idx_upscaled[0]//10)
                    max_coords_c = (max_idx_upscaled[1]//10)
                    ccm_max_value = upscaled_ccm_slice[max_idx_upscaled[0],max_idx_upscaled[1]]

                    print(max_idx,max_coords_r,max_coords_c)
                    print(slice_ccm[max_idx[0],max_idx[1]], ccm_max_value)

                    ccm_cols = slice_ccm.shape[1]
                    ccm_rows = slice_ccm.shape[0]
                    if ccm_max_value >= min_similarity:
                        vector_c = ccm_cols / 2.0 - max_coords_c - 1 
                        vector_r = ccm_rows / 2.0 - max_coords_r - 1
                        flow_arrows[n_flow_arrows,0] = col_start+block_nCols/2.0
                        flow_arrows[n_flow_arrows,1] = row_start+block_nRows/2.0
                        flow_arrows[n_flow_arrows,2] = vector_c
                        flow_arrows[n_flow_arrows,3] = vector_r
                        n_flow_arrows += 1

            if n_flow_arrows==0:
                return None

            translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
            translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)

            max_distance = sqrt(nRows * nRows + nCols * nCols)

            for j in range(nRows):
                for i in range(nCols):
                    # iterate over vectors
                    dx = 0
                    dy = 0 
                    w_sum = 0

                    if n_flow_arrows == 1:
                        dx = flow_arrows[0,2]
                        dy = flow_arrows[0,3]
                    else:

                        distances = np.zeros(n_flow_arrows).astype(np.float32)
                        all_distances = 0
                        for n_arrow in range(n_flow_arrows):
                            d = sqrt(pow(flow_arrows[n_arrow,0] - i, 2) + pow(flow_arrows[n_arrow,1] - j, 2)) + 1
                            distances[n_arrow] = d
                            all_distances += pow(((max_distance - d) / (max_distance * d)), 2)

                        for idx in range(n_flow_arrows):
                            d = distances[idx]
                            first_term = pow(((max_distance - d) / (max_distance * d)), 2)
                            second_term = all_distances

                            weight = first_term / second_term
                            dx += flow_arrows[idx,2] * weight
                            dy += flow_arrows[idx,3] * weight
                            w_sum += weight

                        dx = dx / w_sum
                        dy = dy / w_sum

                    _translation_matrix_c[j, i] = dx
                    _translation_matrix_r[j, i] = dy

            if blocks_per_axis > 1:
                translation_matrix_c = gaussian(translation_matrix_c, sigma=max(block_nCols, block_nRows / 2.0))
                translation_matrix_r = gaussian(translation_matrix_r, sigma=max(block_nCols, block_nRows / 2.0))

            translation_masks[channel,:,:nCols] = _translation_matrix_c
            translation_masks[channel,:, nCols:] = _translation_matrix_r

        return np.array(translation_masks)

    def _run_threaded_dynamic(self, float[:,:,:] img_stack, float[:,:] img_ref, int max_shift, int blocks_per_axis, float min_similarity):

        crsm = CRShiftAndMagnify()
        
        cdef int nChannels =  img_stack.shape[0]    
        cdef int nRows =  img_stack.shape[1]
        cdef int nCols = img_stack.shape[2]    

        cdef int block_nRows = nRows // blocks_per_axis
        cdef int block_nCols = nCols // blocks_per_axis

        cdef float[:,:,:] translation_masks = np.zeros((nChannels, nRows, nCols * 2)).astype(np.float32)
    
        cdef int channel, row_i, col_i
        cdef int row_start, col_start
        cdef int ccm_col_start, ccm_row_start
        cdef float[:,:] img_slice 
        cdef float[:,:] slice_crop, ref_crop, slice_ccm

        cdef float[:,:] flow_arrows = np.zeros((blocks_per_axis**2,4)).astype(np.float32)
        cdef int n_flow_arrows = 0

        cdef int max_coords_r, max_coords_c
        cdef float[:,:] upscaled_ccm_slice
        cdef float vector_c, vector_r, ccm_max_value

        cdef int ccm_cols, ccm_rows
        translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)
        translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
        cdef float[:,:] _translation_matrix_c = translation_matrix_c
        cdef float[:,:] _translation_matrix_r = translation_matrix_r

        cdef int j,i,n_arrow,idx
        cdef float dx,dy,w_sum,max_distance,all_distances
        cdef float[:] distances 
        cdef float d, first_term, second_term, weight

        for channel in range(nChannels):
            img_slice = img_stack[channel,:,:]
            n_flow_arrows = 0
            for row_i in range(blocks_per_axis):
                for col_i in range(blocks_per_axis):
                    row_start = row_i * block_nRows
                    col_start = col_i * block_nCols

                    slice_crop = img_slice[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    ref_crop = img_ref[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    slice_ccm = _calculate_ccm_from_ref(slice_crop, ref_crop)

                    if max_shift > 0 and max_shift * 2 + 1 < slice_ccm.shape[0] and max_shift * 2 + 1 < slice_ccm.shape[1]:
                        ccm_col_start = slice_ccm.shape[1] // 2 - max_shift
                        ccm_row_start = slice_ccm.shape[0] // 2 - max_shift
                        slice_ccm = slice_ccm[ccm_row_start:ccm_row_start+(max_shift*2), ccm_col_start:ccm_col_start + (max_shift * 2)]

                    max_idx = np.unravel_index(np.argmax(slice_ccm), np.array(slice_ccm).shape)
                    # Upscale 10x around max_idx                     
                    upscaled_ccm_slice = crsm.run(np.ascontiguousarray(slice_ccm),0,0,10,10)[0]                    
                    max_idx_upscaled = np.unravel_index(np.argmax(upscaled_ccm_slice), np.array(upscaled_ccm_slice).shape)
                    max_coords_r = (max_idx_upscaled[0]//10)
                    max_coords_c = (max_idx_upscaled[1]//10)
                    ccm_max_value = upscaled_ccm_slice[max_idx_upscaled[0],max_idx_upscaled[1]]

                    print(max_idx,max_coords_r,max_coords_c)
                    print(slice_ccm[max_idx[0],max_idx[1]], ccm_max_value)

                    ccm_cols = slice_ccm.shape[1]
                    ccm_rows = slice_ccm.shape[0]
                    if ccm_max_value >= min_similarity:
                        vector_c = ccm_cols / 2.0 - max_coords_c - 1 
                        vector_r = ccm_rows / 2.0 - max_coords_r - 1
                        flow_arrows[n_flow_arrows,0] = col_start+block_nCols/2.0
                        flow_arrows[n_flow_arrows,1] = row_start+block_nRows/2.0
                        flow_arrows[n_flow_arrows,2] = vector_c
                        flow_arrows[n_flow_arrows,3] = vector_r
                        n_flow_arrows += 1

            if n_flow_arrows==0:
                return None

            translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
            translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)

            max_distance = sqrt(nRows * nRows + nCols * nCols)

            for j in range(nRows):
                for i in range(nCols):
                    # iterate over vectors
                    dx = 0
                    dy = 0 
                    w_sum = 0

                    if n_flow_arrows == 1:
                        dx = flow_arrows[0,2]
                        dy = flow_arrows[0,3]
                    else:

                        distances = np.zeros(n_flow_arrows).astype(np.float32)
                        all_distances = 0
                        for n_arrow in range(n_flow_arrows):
                            d = sqrt(pow(flow_arrows[n_arrow,0] - i, 2) + pow(flow_arrows[n_arrow,1] - j, 2)) + 1
                            distances[n_arrow] = d
                            all_distances += pow(((max_distance - d) / (max_distance * d)), 2)

                        for idx in range(n_flow_arrows):
                            d = distances[idx]
                            first_term = pow(((max_distance - d) / (max_distance * d)), 2)
                            second_term = all_distances

                            weight = first_term / second_term
                            dx += flow_arrows[idx,2] * weight
                            dy += flow_arrows[idx,3] * weight
                            w_sum += weight

                        dx = dx / w_sum
                        dy = dy / w_sum

                    _translation_matrix_c[j, i] = dx
                    _translation_matrix_r[j, i] = dy

            if blocks_per_axis > 1:
                translation_matrix_c = gaussian(translation_matrix_c, sigma=max(block_nCols, block_nRows / 2.0))
                translation_matrix_r = gaussian(translation_matrix_r, sigma=max(block_nCols, block_nRows / 2.0))

            translation_masks[channel,:,:nCols] = _translation_matrix_c
            translation_masks[channel,:, nCols:] = _translation_matrix_r

        return np.array(translation_masks)

    def _run_threaded_static(self, float[:,:,:] img_stack, float[:,:] img_ref, int max_shift, int blocks_per_axis, float min_similarity):

        crsm = CRShiftAndMagnify()
        
        cdef int nChannels =  img_stack.shape[0]    
        cdef int nRows =  img_stack.shape[1]
        cdef int nCols = img_stack.shape[2]    

        cdef int block_nRows = nRows // blocks_per_axis
        cdef int block_nCols = nCols // blocks_per_axis

        cdef float[:,:,:] translation_masks = np.zeros((nChannels, nRows, nCols * 2)).astype(np.float32)
    
        cdef int channel, row_i, col_i
        cdef int row_start, col_start
        cdef int ccm_col_start, ccm_row_start
        cdef float[:,:] img_slice 
        cdef float[:,:] slice_crop, ref_crop, slice_ccm

        cdef float[:,:] flow_arrows = np.zeros((blocks_per_axis**2,4)).astype(np.float32)
        cdef int n_flow_arrows = 0

        cdef int max_coords_r, max_coords_c
        cdef float[:,:] upscaled_ccm_slice
        cdef float vector_c, vector_r, ccm_max_value

        cdef int ccm_cols, ccm_rows
        translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)
        translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
        cdef float[:,:] _translation_matrix_c = translation_matrix_c
        cdef float[:,:] _translation_matrix_r = translation_matrix_r

        cdef int j,i,n_arrow,idx
        cdef float dx,dy,w_sum,max_distance,all_distances
        cdef float[:] distances 
        cdef float d, first_term, second_term, weight

        for channel in range(nChannels):
            img_slice = img_stack[channel,:,:]
            n_flow_arrows = 0
            for row_i in range(blocks_per_axis):
                for col_i in range(blocks_per_axis):
                    row_start = row_i * block_nRows
                    col_start = col_i * block_nCols

                    slice_crop = img_slice[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    ref_crop = img_ref[row_start:row_start+block_nRows, col_start:col_start+block_nCols]
                    slice_ccm = _calculate_ccm_from_ref(slice_crop, ref_crop)

                    if max_shift > 0 and max_shift * 2 + 1 < slice_ccm.shape[0] and max_shift * 2 + 1 < slice_ccm.shape[1]:
                        ccm_col_start = slice_ccm.shape[1] // 2 - max_shift
                        ccm_row_start = slice_ccm.shape[0] // 2 - max_shift
                        slice_ccm = slice_ccm[ccm_row_start:ccm_row_start+(max_shift*2), ccm_col_start:ccm_col_start + (max_shift * 2)]

                    max_idx = np.unravel_index(np.argmax(slice_ccm), np.array(slice_ccm).shape)
                    # Upscale 10x around max_idx                     
                    upscaled_ccm_slice = crsm.run(np.ascontiguousarray(slice_ccm),0,0,10,10)[0]                    
                    max_idx_upscaled = np.unravel_index(np.argmax(upscaled_ccm_slice), np.array(upscaled_ccm_slice).shape)
                    max_coords_r = (max_idx_upscaled[0]//10)
                    max_coords_c = (max_idx_upscaled[1]//10)
                    ccm_max_value = upscaled_ccm_slice[max_idx_upscaled[0],max_idx_upscaled[1]]

                    print(max_idx,max_coords_r,max_coords_c)
                    print(slice_ccm[max_idx[0],max_idx[1]], ccm_max_value)

                    ccm_cols = slice_ccm.shape[1]
                    ccm_rows = slice_ccm.shape[0]
                    if ccm_max_value >= min_similarity:
                        vector_c = ccm_cols / 2.0 - max_coords_c - 1 
                        vector_r = ccm_rows / 2.0 - max_coords_r - 1
                        flow_arrows[n_flow_arrows,0] = col_start+block_nCols/2.0
                        flow_arrows[n_flow_arrows,1] = row_start+block_nRows/2.0
                        flow_arrows[n_flow_arrows,2] = vector_c
                        flow_arrows[n_flow_arrows,3] = vector_r
                        n_flow_arrows += 1

            if n_flow_arrows==0:
                return None

            translation_matrix_r = np.zeros((nRows, nCols)).astype(np.float32)
            translation_matrix_c = np.zeros((nRows, nCols)).astype(np.float32)

            max_distance = sqrt(nRows * nRows + nCols * nCols)

            for j in range(nRows):
                for i in range(nCols):
                    # iterate over vectors
                    dx = 0
                    dy = 0 
                    w_sum = 0

                    if n_flow_arrows == 1:
                        dx = flow_arrows[0,2]
                        dy = flow_arrows[0,3]
                    else:

                        distances = np.zeros(n_flow_arrows).astype(np.float32)
                        all_distances = 0
                        for n_arrow in range(n_flow_arrows):
                            d = sqrt(pow(flow_arrows[n_arrow,0] - i, 2) + pow(flow_arrows[n_arrow,1] - j, 2)) + 1
                            distances[n_arrow] = d
                            all_distances += pow(((max_distance - d) / (max_distance * d)), 2)

                        for idx in range(n_flow_arrows):
                            d = distances[idx]
                            first_term = pow(((max_distance - d) / (max_distance * d)), 2)
                            second_term = all_distances

                            weight = first_term / second_term
                            dx += flow_arrows[idx,2] * weight
                            dy += flow_arrows[idx,3] * weight
                            w_sum += weight

                        dx = dx / w_sum
                        dy = dy / w_sum

                    _translation_matrix_c[j, i] = dx
                    _translation_matrix_r[j, i] = dy

            if blocks_per_axis > 1:
                translation_matrix_c = gaussian(translation_matrix_c, sigma=max(block_nCols, block_nRows / 2.0))
                translation_matrix_r = gaussian(translation_matrix_r, sigma=max(block_nCols, block_nRows / 2.0))

            translation_masks[channel,:,:nCols] = _translation_matrix_c
            translation_masks[channel,:, nCols:] = _translation_matrix_r

        return np.array(translation_masks)


# TODO FINISH ESTIMATOR 
# TODO REFACTOR METHODS.CHANNEL_REGISTRATION TO HOLD THE NEW CLASSES
# TODO TRY TO PARALLELIZE THE SECOND LOOP