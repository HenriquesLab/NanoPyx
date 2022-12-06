from .methods.drift_correction.drift_corrector import DriftCorrector
from .methods.drift_correction.drift_estimator import DriftEstimator
from .methods.channel_alignment.channel_alignment_estimator import ChannelAlignmentEstimator
from .methods.channel_alignment.channel_alignment_corrector import ChannelAlignmentCorrector

#TODO write tests, create simulated data for tests and write docstrings

def estimate_drift_correction(image_array, save_as_npy=True, save_drift_table_path=None, roi=None, **kwargs):
    estimator = DriftEstimator()
    corrected_img = estimator.estimate(image_array, roi=roi, **kwargs)
    print(save_drift_table_path)
    estimator.save_drift_table(save_as_npy=save_as_npy, path=save_drift_table_path)
    if corrected_img is not None:
        return corrected_img
    else:
        pass

def apply_drift_correction(image_array, path=None, drift_table=None):
    corrector = DriftCorrector()
    if drift_table is None:
        corrector.load_drift_table(path=path)
    else:
        corrector.estimator_table = drift_table
    corrected_img = corrector.apply_correction(image_array)

    if corrected_img is not None:
        return corrected_img
    else:
        print("No corrected image was generated")
        pass

def estimate_channel_alignment(image_array, ref_channel, max_shift, blocks_per_axis, min_similarity, method="subpixel",
                               save_translation_masks=True, translation_mask_save_path=None,
                               save_ccms=False, ccms_save_path=False, apply=True):
    estimator = ChannelAlignmentEstimator()
    aligned_image = estimator.estimate(image_array, ref_channel, max_shift, blocks_per_axis, min_similarity, method=method,
                                       save_translation_masks=save_translation_masks, translation_mask_save_path=translation_mask_save_path,
                                       save_ccms=save_ccms, ccms_save_path=ccms_save_path, apply=apply)

    if aligned_image is not None:
        return aligned_image
    else:
        pass

def apply_channel_alignment(image_array, translation_masks=None):
    corrector = ChannelAlignmentCorrector()
    aligned_image = corrector.align_channels(image_array, translation_masks=translation_masks)

    if aligned_image is not None:
        return aligned_image
    else:
        print("No aligned image was generated")
        pass
