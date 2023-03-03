import numpy as np
from skimage.io import imread

from ...core.transform.interpolation_catmull_rom import interpolate


class ChannelRegistrationCorrector(object):
    def __init__(self):
        self.aligned_stack = None

    def load_translation_masks(self, path=None):
        if path is not None:
            path = input("Please provide a filepath to the translation masks")

        return imread(path)

    def align_channels(self, img_stack, translation_masks=None):
        
        translation_masks = translation_masks

        if translation_masks is None:
            translation_masks = self.load_translation_masks()

        input_d_type = img_stack.dtype

        n_channels = img_stack.shape[0]
        height = img_stack.shape[1]
        width = img_stack.shape[2]

        self.aligned_stack = np.empty((n_channels, height, width))
        channels_list = list(range(n_channels))

        for channel in channels_list:
            img_slice = img_stack[channel].astype(np.float32)
            translation_mask = translation_masks[channel]
            if np.sum(translation_mask) == 0:
                self.aligned_stack[channel] = img_slice
            else:
                for y_i in range(height):
                    for x_i in range(width):
                        dx = translation_mask[y_i, x_i]
                        dy = translation_mask[y_i, x_i + width]
                        value = interpolate(img_slice, x_i-dx, y_i-dy)
                        self.aligned_stack[channel][y_i, x_i] = value

        return self.aligned_stack.astype(input_d_type)

