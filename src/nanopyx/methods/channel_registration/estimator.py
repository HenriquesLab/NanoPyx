import numpy as np
from numpy import array
from skimage.io import imsave

from .corrector import ChannelRegistrationCorrector
from ...core.analysis.cross_correlation_elastic import calculate_translation_mask


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
        save_ccms(path): Save cross-correlation matrices to a file.
        estimate(img_stack, ref_channel, max_shift, blocks_per_axis, min_similarity, method, save_translation_masks, translation_mask_save_path, save_ccms, ccms_save_path, algorithm, apply): Estimate and apply channel registration.

    Note:
        This class is designed for estimating and applying channel registration to an image stack.
        It calculates translation masks for channel alignment and provides options for saving results to files.
    """

    def __init__(self) -> None:
        """
        Initialize a ChannelRegistrationEstimator instance.
        """
        self.translation_masks = None
        self.ccms = None

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
        return corrector.align_channels(img_stack, translation_masks=self.translation_masks)

    def calculate_translation(
        self,
        channel_to_align,
        ref_channel_img,
        max_shift,
        blocks_per_axis,
        min_similarity,
        algorithm="field",
        method="subpixel",
    ):
        """
        Calculate a translation mask for aligning a channel with a reference channel image.

        Args:
            channel_to_align (numpy.ndarray): The channel to be aligned.
            ref_channel_img (numpy.ndarray): The reference channel image.
            max_shift (float): Maximum shift allowed for alignment.
            blocks_per_axis (int): Number of blocks per axis for cross-correlation.
            min_similarity (float): Minimum similarity threshold for alignment.
            algorithm (str, optional): Translation mask interpolation algorithm to use (default is "field", "weight" is the other option).
            method (str, optional): Subpixel method for alignment (default is "subpixel").

        Returns:
            numpy.ndarray: The calculated translation mask.

        Example:
            estimator = ChannelRegistrationEstimator()
            translation_mask = estimator.calculate_translation(
                channel_to_align, ref_channel_img, max_shift, blocks_per_axis, min_similarity
            )

        Note:
            This method calculates a translation mask for aligning a channel with a reference channel image.
            It uses the provided parameters for alignment, such as maximum shift, block configuration,
            and minimum similarity threshold, along with optional parameters for the alignment algorithm
            and subpixel method.
        """
        translation_mask = calculate_translation_mask(
            channel_to_align,
            ref_channel_img,
            max_shift,
            blocks_per_axis,
            min_similarity,
            algorithm=algorithm,
            method=method,
        )
        return translation_mask

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
            path = input("Please provide a filepath to save the translation masks") + "_translation_masks.tif"

        imsave(path + "_translation_masks.tif", self.translation_masks)

    def save_ccms(self, path=None):
        """
        Save the cross-correlation matrices (ccms) to a file.

        Args:
            path (str, optional): The file path to save the ccms.
                If not provided, a user input prompt will be used to specify the path.
                The default file name will be "_ccms.tif" appended to the specified path.

        Example:
            estimator = ChannelRegistrationEstimator()
            estimator.save_ccms("ccms.tif")

        Note:
            This method saves the cross-correlation matrices (ccms) to a TIFF file.
            If the `path` argument is not provided, it prompts the user to input a file path
            and appends "_ccms.tif" to it as the default file name.
        """
        if path is None:
            path = input("Please provide a filepath to save the ccms") + "_ccms.tif"

        imsave(path + "_ccms.tif", self.ccms)

    def estimate(
        self,
        img_stack: array,
        ref_channel: int,
        max_shift: float,
        blocks_per_axis: int,
        min_similarity: float,
        method: str = "subpixel",
        save_translation_masks: bool = True,
        translation_mask_save_path: str = None,
        save_ccms: bool = False,
        ccms_save_path: str = None,
        algorithm: str = "field",
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
            method (str, optional): Subpixel method for alignment (default is "subpixel").
            save_translation_masks (bool, optional): Whether to save translation masks (default is True).
            translation_mask_save_path (str, optional): The file path to save translation masks.
                If not provided, a user input prompt will be used to specify the path.
            save_ccms (bool, optional): Whether to save cross-correlation matrices (ccms) (default is False).
            ccms_save_path (str, optional): The file path to save ccms.
                If not provided, a user input prompt will be used to specify the path.
            algorithm (str, optional): Cross-correlation algorithm to use (default is "field").
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
        channels_to_align = list(range(img_stack.shape[0]))
        channels_to_align.remove(ref_channel)

        if ref_channel > img_stack.shape[1]:
            print("Reference channel number cannot be bigger than number of channels!")
            return None

        self.translation_masks = np.zeros((img_stack.shape[0], img_stack.shape[1], img_stack.shape[2] * 2))
        self.ccms = []

        for channel in channels_to_align:
            translation_mask, ccm = self.calculate_translation(
                img_stack[channel],
                img_stack[ref_channel],
                max_shift,
                blocks_per_axis,
                min_similarity,
                algorithm=algorithm,
                method=method,
            )
            self.translation_masks[channel] = translation_mask
            self.ccms.append(ccm)

        self.ccms.insert(ref_channel, np.zeros((len(self.ccms[0]), len(self.ccms[0][0]))))
        self.ccms = np.array(self.ccms)

        if save_translation_masks:
            self.save_translation_mask(path=translation_mask_save_path)

        if save_ccms:
            self.save_ccms(path=ccms_save_path)

        if apply:
            return self.apply_elastic_transform(img_stack)
        else:
            return None
