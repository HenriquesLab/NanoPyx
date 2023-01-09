import nanopyx

def test_channel_registration(random_channel_misalignment):
    estimator = nanopyx.methods.channel_registration.estimator.ChannelRegistrationEstimator()
    aligned_img = estimator.estimate(random_channel_misalignment, 0, 200, 3, 0.5, save_ccms=False, ccms_save_path="",
                                     save_translation_masks=False, translation_mask_save_path="", apply=True)

    drift_estimator = nanopyx.methods.drift_alignment.estimator.DriftEstimator()
    drift_estimator.estimate(aligned_img, ref_option=0, apply=False)
    drift_table = drift_estimator.estimator_table.drift_table

    assert drift_table[0, 0] == 0
