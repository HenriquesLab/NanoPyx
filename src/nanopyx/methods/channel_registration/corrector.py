from gettext import translation
import numpy as np
from skimage.io import imread
from scipy.interpolate import griddata, interp2d

from ...core.utils.time.timeit import timeit


class ChannelRegistrationCorrector(object):
    def __init__(self):
        self.aligned_stack = None

    def load_translation_masks(self, path=None):
        if path is not None:
            return imread(path)
        else:
            print("Please provide a filename path to load translation masks")

    def align_channels(self, img_stack, translation_masks=None):
        
        translation_masks = translation_masks

        if translation_masks is None:
            print("Please pass translation masks as an argument")
        else:
            n_channels = img_stack.shape[0]
            height = img_stack.shape[1]
            width = img_stack.shape[2]

            self.aligned_stack = np.zeros((n_channels, height, width))
            channels_list = list(range(n_channels))

            for channel in channels_list:
                img_slice = img_stack[channel]
                translation_mask = translation_masks[channel]
                if np.sum(translation_mask) == 0:
                    self.aligned_stack[channel] = img_slice
                else:
                    x = [xi for xi in range(img_slice.shape[1])]
                    y = [yi for yi in range(img_slice.shape[0])]
                    z = img_slice.reshape((img_slice.shape[0] * img_slice.shape[1]))
                    interpolator = interp2d(y, x, z, kind="cubic")
                    for y_i in range(height):
                        for x_i in range(width):
                            dx = translation_mask[y_i, x_i]
                            dy = translation_mask[y_i, x_i + width]
                            value = interpolator(y_i-dy, x_i-dx)
                            self.aligned_stack[channel][y_i, x_i] = value


            return self.aligned_stack

