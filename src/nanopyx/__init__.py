from .methods.drift_alignment.corrector import DriftCorrector
from .methods.drift_alignment.estimator import DriftEstimator
from .methods.channel_registration.estimator import ChannelRegistrationEstimator
from .methods.channel_registration.corrector import ChannelRegistrationCorrector
from .core.utils.time.timeit import timeit

#TODO write docstrings for all methods and generate documentation with sphynx

@timeit
def estimate_drift_alignment(image_array, save_as_npy=True, save_drift_table_path=None, roi=None, **kwargs):
    estimator = DriftEstimator()
    corrected_img = estimator.estimate(image_array, roi=roi, **kwargs)
    print(save_drift_table_path)
    estimator.save_drift_table(save_as_npy=save_as_npy, path=save_drift_table_path)
    if corrected_img is not None:
        return corrected_img
    else:
        pass

@timeit
def apply_drift_alignment(image_array, path=None, drift_table=None):
    corrector = DriftCorrector()
    if drift_table is None:
        corrector.load_estimator_table(path=path)
    else:
        corrector.estimator_table = drift_table
    corrected_img = corrector.apply_correction(image_array)
    return corrected_img

@timeit
def estimate_channel_registration(image_array, ref_channel, max_shift, blocks_per_axis, min_similarity, method="subpixel",
                               save_translation_masks=True, translation_mask_save_path=None,
                               save_ccms=False, ccms_save_path=False, apply=True):
    estimator = ChannelRegistrationEstimator()
    aligned_image = estimator.estimate(image_array, ref_channel, max_shift, blocks_per_axis, min_similarity, method=method,
                                       save_translation_masks=save_translation_masks, translation_mask_save_path=translation_mask_save_path,
                                       save_ccms=save_ccms, ccms_save_path=ccms_save_path, apply=apply)

    if aligned_image is not None:
        return aligned_image
    else:
        pass

@timeit
def apply_channel_registration(image_array, translation_masks=None):
    corrector = ChannelRegistrationCorrector()
    aligned_image = corrector.align_channels(image_array, translation_masks=translation_masks)

    return aligned_image
