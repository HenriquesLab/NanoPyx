from gettext import translation
import numpy as np
from skimage.io import imread
from tkinter import filedialog as fd
from scipy.interpolate import griddata, interp2d

from enanoscopy.methods.utils.timeit import timeit


class ChannelAlignmentCorrector(object):
    def __init__(self):
        self.aligned_stack = None
        self.translation_masks = None

    def load_translation_masks(self, path=None):
        if path is not None:
            self.translation_masks = imread(path)
        else:
            self.translation_masks = imread(fd.askopenfilename(title="Load Translation Masks"))

    @timeit
    def get_interpolated_value(self, img_slice, coords):
        points = [(y, x) for y in range(img_slice.shape[0]) for x in range(img_slice.shape[1])]
        values = img_slice.reshape((img_slice.shape[0] * img_slice.shape[1]))

        return griddata(points, values, coords, method="cubic")

    @timeit
    def align_channels(self, img_stack, translation_masks=None):
        
        self.translation_masks = translation_masks

        if self.translation_masks is None:
            self.load_translation_masks()

        n_channels = img_stack.shape[0]
        height = img_stack.shape[1]
        width = img_stack.shape[2]

        self.aligned_stack = np.zeros((n_channels, height, width))
        channels_list = list(range(n_channels))

        for channel in channels_list:
            img_slice = img_stack[channel]
            x = [xi for xi in range(img_slice.shape[1])]
            y = [yi for yi in range(img_slice.shape[0])]
            z = img_slice.reshape((img_slice.shape[0] * img_slice.shape[1]))
            interpolator = interp2d(y, x, z, kind="cubic") # this is much faster than using griddata
            if np.sum(translation_masks[channel]) == 0:
                self.aligned_stack[channel] = img_stack[channel]
            else:
                for y_i in range(height):
                    for x_i in range(width):
                        dx = self.translation_masks[channel][y_i, x_i]
                        dy = self.translation_masks[channel][y_i, x_i + width]
                        #value = self.get_interpolated_value(img_stack[channel], (y_i+dy, x_i+dx))
                        value = interpolator(dy, dx)
                        self.aligned_stack[channel][y_i, x_i] = value

        #TODO fix bug translation masks

        return self.aligned_stack

