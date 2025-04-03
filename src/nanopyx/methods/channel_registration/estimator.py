import numpy as np
from numpy import array
from skimage.io import imsave

from .corrector import ChannelRegistrationCorrector
from ...core.analysis._le_channel_registration import (
    ChannelRegistrationEstimator as leChannelRegistrationEstimator,
)


# this class assumes that the image is a numpy array with shape = [n_channels, height, width]
# assumes that channels in an image that will be aligned using generated translation masks will be in the same order
class ChannelRegistrationEstimator(object):
    """
    A class for estimating and applying channel registration to an image stack.

    This class assumes that the image is a numpy array with shape [n_channels, height, width].
    It also assumes that the channels in an image that will be aligned using generated translation masks
    will be in the same order.

    Attributes:
        translation_masks (numpy.ndarray): An array to store translation masks.
        ccms (numpy.ndarray): An array to store cross-correlation matrices.

    Methods:
        apply_elastic_transform(img_stack): Apply elastic transformations to align channels.
        calculate_translation(channel_to_align, ref_channel_img, max_shift, blocks_per_axis, min_similarity, algorithm, method): Calculate translation masks for channel alignment.
        save_translation_mask(path): Save translation masks to a file.
        estimate(img_stack, ref_channel, max_shift, blocks_per_axis, min_similarity, method, save_translation_masks, translation_mask_save_path, algorithm, apply): Estimate and apply channel registration.

    Note:
        This class is designed for estimating and applying channel registration to an image stack.
        It calculates translation masks for channel alignment and provides options for saving results to files.
    """

    def __init__(self, verbose=True) -> None:
        """
        Initialize a ChannelRegistrationEstimator instance.
        """
        self.verbose = verbose
        self.translation_masks = None

    def apply_elastic_transform(self, img_stack):
        """
        Apply elastic transformations to align channels in an image stack.

        Args:
            img_stack (numpy.ndarray): The image stack with shape [n_channels, height, width].

        Returns:
            numpy.ndarray: The aligned image stack.

        Example:
            estimator = ChannelRegistrationEstimator()
            aligned_stack = estimator.apply_elastic_transform(img_stack)

        Note:
            This method uses the ChannelRegistrationCorrector to apply elastic transformations
            for aligning the channels in the provided image stack based on the stored translation masks.
        """
        corrector = ChannelRegistrationCorrector()
        return corrector.align_channels(
            img_stack, translation_masks=self.translation_masks
        )

    def save_translation_mask(self, path=None):
        """
        Save the translation masks to a file.

        Args:
            path (str, optional): The file path to save the translation masks.
                If not provided, a user input prompt will be used to specify the path.
                The default file name will be "_translation_masks.tif" appended to the specified path.

        Example:
            estimator = ChannelRegistrationEstimator()
            estimator.save_translation_mask("translation_masks.tif")

        Note:
            This method saves the translation masks to a TIFF file. If the `path` argument is not provided,
            it prompts the user to input a file path and appends "_translation_masks.tif" to it as the default file name.
        """
        if path is None:
            path = (
                input(
                    "Please provide a filepath to save the translation masks"
                )
                + "_translation_masks.tif"
            )

        imsave(path + "_translation_masks.tif", self.translation_masks)

    def estimate(
        self,
        img_stack: array,
        ref_channel: int,
        max_shift: float,
        blocks_per_axis: int,
        min_similarity: float,
        save_translation_masks: bool = True,
        translation_mask_save_path: str = None,
        apply: bool = False,
    ):
        """
        Estimate and perform channel registration on an image stack.

        Args:
            img_stack (numpy.ndarray): The image stack with shape [n_channels, height, width].
            ref_channel (int): The reference channel index for alignment.
            max_shift (float): Maximum shift allowed for alignment.
            blocks_per_axis (int): Number of blocks per axis for cross-correlation.
            min_similarity (float): Minimum similarity threshold for alignment.
            save_translation_masks (bool, optional): Whether to save translation masks (default is True).
            translation_mask_save_path (str, optional): The file path to save translation masks.
                If not provided, a user input prompt will be used to specify the path.
            apply (bool, optional): Whether to apply elastic transformation to the image stack (default is False).

        Returns:
            numpy.ndarray or None: If `apply` is True, returns the aligned image stack; otherwise, returns None.

        Example:
            estimator = ChannelRegistrationEstimator()
            aligned_stack = estimator.estimate(
                img_stack, ref_channel=0, max_shift=1.0, blocks_per_axis=32, min_similarity=0.5
            )

        Note:
            This method estimates channel registration for aligning channels in the provided image stack.
            It calculates translation masks and cross-correlation matrices (ccms) for alignment.
            The alignment is performed based on the specified parameters, and the results can be optionally saved.
            If `apply` is True, it applies elastic transformation to the image stack and returns the aligned stack.
        """

        if ref_channel > img_stack.shape[0]:
            print(
                "Reference channel number cannot be bigger than number of channels!"
            )
            return None

        estimator = leChannelRegistrationEstimator(verbose=self.verbose)
        self.translation_masks = estimator.run(
            np.asarray(img_stack, dtype=np.float32),
            ref_channel,
            max_shift,
            blocks_per_axis,
            min_similarity,
        )

        if save_translation_masks:
            self.save_translation_mask(path=translation_mask_save_path)

        if apply:
            return self.apply_elastic_transform(img_stack)
        else:
            return None
