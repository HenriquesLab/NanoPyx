import numpy as np
from skimage.io import imread

# TODO recheck values
from ...core.transform.align_channels import align_channels as new_align_channels


class ChannelRegistrationCorrector(object):
    """
    Corrector class for channel registration in an image stack.

    This class provides methods for aligning channels in an image stack using translation masks.

    Args:
        None

    Attributes:
        aligned_stack (numpy.ndarray): The aligned image stack after correction.

    Methods:
        load_translation_masks(path=None): Load translation masks from a file.

        align_channels(img_stack, translation_masks=None): Align channels in an image stack using translation masks.

    Example:
        corrector = ChannelRegistrationCorrector()
        translation_masks = corrector.load_translation_masks("translation_masks.tif")
        aligned_stack = corrector.align_channels(img_stack, translation_masks)

    Note:
        The `ChannelRegistrationCorrector` class is used for correcting channel registration in an image stack.
        It provides methods for loading translation masks and aligning channels using these masks.
    """

    def __init__(self):
        """
        Initialize the `ChannelRegistrationCorrector` object.

        Args:
            None

        Returns:
            None

        Example:
            corrector = ChannelRegistrationCorrector()
        """
        self.aligned_stack = None

    def load_translation_masks(self, path=None):
        """
        Load translation masks from a file.

        Args:
            path (str, optional): The file path to load translation masks from. If not provided, a user input prompt will be used to specify the path.

        Returns:
            numpy.ndarray: The loaded translation masks as a NumPy array.

        Example:
            translation_masks = corrector.load_translation_masks("translation_masks.tif")

        Note:
            This method loads translation masks from a file and returns them as a NumPy array.
        """
        if path is not None:
            path = input("Please provide a filepath to the translation masks")

        return imread(path)

    def align_channels(self, img_stack, translation_masks=None):
        """
        Align channels in an image stack using translation masks.

        Args:
            img_stack (numpy.ndarray): The input image stack with shape [n_channels, height, width].
            translation_masks (numpy.ndarray, optional): The translation masks to use for alignment. If not provided, they will be loaded.

        Returns:
            numpy.ndarray: The aligned image stack after correction.

        Example:
            aligned_stack = corrector.align_channels(img_stack, translation_masks)

        Note:
            This method aligns channels in an image stack using translation masks.
        """
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
            translation_mask = translation_masks[channel].astype(np.float32)
            if np.sum(translation_mask) == 0:
                self.aligned_stack[channel] = img_slice
            else:
                self.aligned_stack[channel] = new_align_channels(np.ascontiguousarray(img_slice.reshape((1, height, width)), dtype=np.float32), translation_mask)

        return self.aligned_stack.astype(input_d_type)
