import multiprocessing as mp
mp.set_start_method("spawn", force=True) #FIXME workaround for macOs to be able to run it from napari
from .methods.drift_correction.drift_corrector import DriftCorrector
from .methods.drift_correction.drift_estimator import DriftEstimator
from .methods.channel_alignment.channel_alignment_estimator import ChannelAlignmentEstimator
from .methods.channel_alignment.channel_alignment_corrector import ChannelAlignmentCorrector
# this library assumes image_array is an array with shape as defined in numpy arrays (time, y, x) or (time, y, x, z)

#TODO write tests for shift from ccm and alignment correction, and test differences between linear and cubic interpolation

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
    corrected_img = corrector.apply_correction(image_array)

    if corrected_img is not None:
        return corrected_img
    else:
        print("No corrected image was generated")
        pass

def estimate_channel_alignment(image_array, ref_channel, max_shift, blocks_per_axis, min_similarity, apply):
    estimator = ChannelAlignmentEstimator()
    aligned_image = estimator.estimate(image_array, ref_channel, max_shift, blocks_per_axis, min_similarity, apply)

    if aligned_image is not None:
        return aligned_image
    else:
        pass