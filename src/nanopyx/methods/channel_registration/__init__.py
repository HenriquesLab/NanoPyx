from .estimator import ChannelRegistrationEstimator
from .corrector import ChannelRegistrationCorrector
from ...core.utils.timeit import timeit


def estimate_channel_registration(
    image_array,
    ref_channel,
    max_shift,
    blocks_per_axis,
    min_similarity,
    save_translation_masks=True,
    translation_mask_save_path=None,
    apply=True,
):
    """
    Function used to estimate shift between different color channels and align them of an image based on cross correlation.
    :param image_array:numpy array  with shape (n_channels, y, x); image to be corrected
    :param ref_channel: int; channel index to be used as reference
    :param max_shift: int; maximum shift accepted for correction, in pixels.
    :param blocks_per_axis: int; number of blocks to divide the image in both x and y dimensions
    :param min_similarity: float; minimum value of similarity to accept a shift as a correction
    subpixel precision, max simply takes the maximum of the cross correlation map
    :param save_translation_masks: bool, defaults to True; whether to save translation masks as a tif or not
    :param translation_mask_save_path: str; path where to save translation masks
    :param apply: bool; whether to apply the correction if True or only estimate if False
    :return: if apply==True, returns corrected image with shape (c, y, x)
    """
    estimator = ChannelRegistrationEstimator()
    aligned_image = estimator.estimate(
        image_array,
        ref_channel,
        max_shift,
        blocks_per_axis,
        min_similarity,
        save_translation_masks=save_translation_masks,
        translation_mask_save_path=translation_mask_save_path,
        apply=apply,
    )

    if aligned_image is not None:
        return aligned_image
    else:
        pass


def apply_channel_registration(image_array, translation_masks=None):
    """
    Function used to align different color channels of an image based on cross correlation.
    :param image_array: numpy array with shape (n_channels, y, x); image to be registered
    :param translation_masks: numpy array of translation masks
    :return: returns corrected image with shape (c, y, x)
    """
    corrector = ChannelRegistrationCorrector()
    aligned_image = corrector.align_channels(image_array, translation_masks=translation_masks)

    return aligned_image
