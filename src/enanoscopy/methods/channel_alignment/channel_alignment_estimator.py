import numpy as np
from numpy import array

from .channel_alignment_corrector import ChannelAlignmentCorrector


#this class assumes that the image is a numpy array with shape = [n_slices, n_channels, height, width]
class ChannelAlignmentEstimator(object):

    def __init__(self) -> None:
        pass

    def estimate(self, img_stack: array, ref_channel: int, max_shift: float, blocks_per_axis: int, min_similarity: float, guassian_blur_radius: int, apply: bool):

        channels_list = range(img_stack.shape[1])
        channels_to_align = channels_list.remove(ref_channel-1)

        if apply:
            realigned_stack = np.array([])

        if ref_channel > img_stack.shape[1]:
            print("Reference channel number cannot be bigger than number of channels!")
            return None

