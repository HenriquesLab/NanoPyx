from .fft_helper_functions import make_even_square, check_even_square, flip
from ...image.analysis.image_statistics import calculate_ppmcc

import numpy as np
import multiprocessing as mp
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

class CrossCorrelationMap(object):
    def __init__(self) -> None:
        self.ccm = []
        self.img_ref = None
        self.img_stack = None
        self.normalize = True

    def _calculate_ccm(self, index):

        img_slice = self.img_stack[index]

        if self.img_ref is None:
            img_ref = self.img_stack[max(index-1, 0)]
        else:
            img_ref = self.img_ref

        image_product = np.fft.fft2(img_ref) * np.fft.fft2(img_slice).conj()
        slice_ccm = np.fft.ifft2(image_product)
        slice_ccm = np.fft.fftshift(slice_ccm).real
        slice_ccm = slice_ccm[::-1, ::-1]
        #slice_ccm = flip(slice_ccm)

        if self.normalize:
            slice_ccm = self._normalize_ccm(img_ref, img_slice, slice_ccm)
        return slice_ccm[0:slice_ccm.shape[0]-1, 0:slice_ccm.shape[1]-1]

    def _normalize_ccm(self, img_ref, img_curr, slice_ccm):
        
        w = slice_ccm.shape[1]
        h = slice_ccm.shape[0]

        ccm_pixels = slice_ccm.reshape(slice_ccm.shape[0] * slice_ccm.shape[1])
        min_idx = np.argmin(ccm_pixels)
        min_value = ccm_pixels[min_idx]
        max_idx = np.argmax(ccm_pixels)
        max_value = ccm_pixels[max_idx]

        shift_x_max = int((max_idx % w) - w / 2)
        shift_y_max = int((max_idx / h) - h / 2)
        shift_x_min = int((min_idx % w) - w / 2)
        shift_y_min = int((min_idx / h) - h / 2)

        max_ppmcc = calculate_ppmcc(img_ref, img_curr, shift_x_max, shift_y_max)
        min_ppmcc = calculate_ppmcc(img_ref, img_curr, shift_x_min, shift_y_min)

        delta_v = max_value - min_value
        #delta_idx = max_ppmcc - min_ppmcc #not currently in use, older implementation?

        for i in range(ccm_pixels.shape[0]):
            value = (ccm_pixels[i] - min_value) / delta_v
            value = (value * (max_ppmcc - min_ppmcc)) + min_ppmcc
            ccm_pixels[i] = value.real

        return ccm_pixels.reshape((slice_ccm.shape[0], slice_ccm.shape[1]))

    def calculate_ccm(self, img_ref, img_stack, normalize):

        self.normalize = normalize

        if img_ref is not None:
            ref_width = img_ref.shape[1]
            ref_height = img_ref.shape[0]
            stack_width = img_stack.shape[2]
            stack_height = img_stack.shape[1]

            if ref_width != stack_width and ref_height != stack_height:
                print("Reference image and image stack do not have the same size!")
                return None

            self.img_ref = make_even_square(img_ref)
        else:
            self.img_ref = None

        if not check_even_square(img_stack):
            tmp_stack = []
            for i in range(img_stack.shape[0]):
                tmp_stack.append(make_even_square(img_stack[i]))

            self.img_stack = np.array(tmp_stack)
        else:
            self.img_stack = img_stack

        ccm = []
        pool = mp.Pool(mp.cpu_count())
        ccm_pool = pool.map_async(self._calculate_ccm, range(0, self.img_stack.shape[0], 1), callback=ccm.append)
        ccm_pool.wait()
        pool.close() # makes pool no longer usable
        #pool.join() # used to catch exceptions, requires expanding functionality
        ccm = np.array(ccm).reshape((self.img_stack.shape[0], self.img_stack.shape[1]-1, self.img_stack.shape[2]-1))
        #ccm = [self._calculate_ccm(i) for i in range(0, self.img_stack.shape[0], 1)]
        ccm = np.array(ccm)

        return ccm


