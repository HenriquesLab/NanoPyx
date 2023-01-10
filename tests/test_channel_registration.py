import nanopyx

def test_channel_registration(random_channel_misalignment):
    estimator = nanopyx.methods.channel_registration.estimator.ChannelRegistrationEstimator()
    aligned_img = estimator.estimate(random_channel_misalignment, 0, 200, 3, 0.5, save_ccms=False, ccms_save_path="",
                                     save_translation_masks=False, translation_mask_save_path="", apply=True)

    drift_estimator = nanopyx.methods.drift_alignment.estimator.DriftEstimator()
    drift_estimator.estimate(aligned_img, ref_option=0, apply=False)
    drift_table = drift_estimator.estimator_table.drift_table

    assert drift_table[0, 0] == 0

def test_channel_registration_init(random_channel_misalignment):
    aligned_img = nanopyx.estimate_channel_registration(random_channel_misalignment, 0, 200, 3, 0.5,
                                                        save_ccms=False, ccms_save_path="",
                                                        save_translation_masks=False, translation_mask_save_path="")
    aligned_tmp = nanopyx.estimate_channel_registration(random_channel_misalignment, 0, 200, 3, 0.5,
                                                        save_ccms=False, ccms_save_path="",
                                                        save_translation_masks=False, translation_mask_save_path="",
                                                        apply=False)
    drift_estimator = nanopyx.methods.drift_alignment.estimator.DriftEstimator()
    drift_estimator.estimate(aligned_img, ref_option=0, apply=False)
    drift_table = drift_estimator.estimator_table.drift_table

    assert drift_table[0, 0] == 0

def test_channel_registration_apply_init(random_channel_misalignment):
    channel_registrator = nanopyx.methods.channel_registration.estimator.ChannelRegistrationEstimator()
    aligned_image = channel_registrator.estimate(random_channel_misalignment, 0, 200, 3, 0.5, save_ccms=False, ccms_save_path="",
                                 save_translation_masks=False, translation_mask_save_path="", apply=True)
    aligned_image_2 = nanopyx.apply_channel_registration(random_channel_misalignment,
                                                       translation_masks=channel_registrator.translation_masks)

    assert (aligned_image==aligned_image_2).all()
