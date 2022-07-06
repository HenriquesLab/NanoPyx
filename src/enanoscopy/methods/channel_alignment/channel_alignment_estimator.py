import numpy as np
from numpy import array
from skimage.io import imsave
from tkinter import filedialog as fd

from .channel_alignment_corrector import ChannelAlignmentCorrector
from ..image.registration.cross_correlation_elastic import calculate_translation_mask


# this class assumes that the image is a numpy array with shape = [n_channels, height, width]
# assumes that channels in an image that will be aligned using generated translation masks will be in the same order
class ChannelAlignmentEstimator(object):

    def __init__(self) -> None:
        self.translation_masks = None

    def apply_elastic_transform(self, img_stack):
        corrector = ChannelAlignmentCorrector()
        return corrector.align_channels(img_stack, self.translation_masks)

    def calculate_translation(self, channel_to_align, ref_channel_img, max_shift,
                                   blocks_per_axis, min_similarity):
        translation_mask = calculate_translation_mask(channel_to_align, ref_channel_img,
                                                      max_shift, blocks_per_axis,
                                                      min_similarity)
        return translation_mask

    def save_translation_mask(self, path=None):
        if path is not None:
            imsave(path + ".tif", self.translation_masks)
        else:
            imsave(fd.asksaveasfilename() + ".tif", self.translation_masks)

    def estimate(self, img_stack: array, ref_channel: int, max_shift: float,
                 blocks_per_axis: int, min_similarity: float, apply: bool):

        channels_to_align = list(range(img_stack.shape[0]))
        channels_to_align.remove(ref_channel)

        if ref_channel > img_stack.shape[1]:
            print("Reference channel number cannot be bigger than number of channels!")
            return None

        self.translation_masks = np.zeros((img_stack.shape[0], img_stack.shape[1]), img_stack.shape[2])

        for channel in channels_to_align:
            translation_mask = self.calculate_translation(img_stack[channel], img_stack[ref_channel], max_shift,
                                                               blocks_per_axis, min_similarity)
            self.translation_masks[channel] = translation_mask

        self.save_translation_mask()

        if apply:
            return self.apply_elastic_transform(img_stack)
        else:
            return None


