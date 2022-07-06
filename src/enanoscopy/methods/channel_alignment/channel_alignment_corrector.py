import numpy as np
from skimage.io import imread
from tkinter import filedialog as fd


class ChannelAlignmentCorrector(object):
    def __init__(self):
        self.aligned_stack = None

    def align_channels(self, img_stack, translation_masks):
        n_channels = img_stack.shape[0]
        height = img_stack.shape[1]
        width = img_stack.shape[2]

        self.aligned_stack = np.zeros(n_channels, height, width)
        channels_list = list(range(n_channels))

        for channel in channels_list:
            if np.sum(translation_masks[channel]) == 0:
                self.aligned_stack[channel] = img_stack[channel]
            else:
                pass
        return self.aligned_stack

    def align_channels_from_previous(self, img_stack, translation_masks_path=None):
        if translation_masks_path is not None:
            return self.align_channels(img_stack, imread(translation_masks_path))
        else:
            return self.align_channels(img_stack, imread(fd.askopenfilename()))

