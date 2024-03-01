from nanopyx.methods import drift_alignment, channel_registration
from nanopyx.methods.channel_registration import estimate_channel_registration, apply_channel_registration

def test_channel_registration(random_channel_misalignment):
    estimator = channel_registration.estimator.ChannelRegistrationEstimator()
    aligned_img = estimator.estimate(random_channel_misalignment, 0, 200, 3, 0.5, save_translation_masks=False, translation_mask_save_path="", apply=True)

    drift_estimator = drift_alignment.estimator.DriftEstimator()
    drift_estimator.estimate(aligned_img, ref_option=0, apply=False)
    drift_table = drift_estimator.estimator_table.drift_table

    assert drift_table[0, 0] == 0

def test_channel_registration_init(random_channel_misalignment):
    aligned_img = estimate_channel_registration(random_channel_misalignment, 0, 200, 3, 0.5, save_translation_masks=False,
                                                translation_mask_save_path="")
    aligned_tmp = estimate_channel_registration(random_channel_misalignment, 0, 200, 3, 0.5, save_translation_masks=False,
                                                translation_mask_save_path="", apply=False)
    drift_estimator = drift_alignment.estimator.DriftEstimator()
    drift_estimator.estimate(aligned_img, ref_option=0, apply=False)
    drift_table = drift_estimator.estimator_table.drift_table

    assert drift_table[0, 0] == 0

def test_channel_registration_apply_init(random_channel_misalignment):
    channel_registrator = channel_registration.estimator.ChannelRegistrationEstimator()
    aligned_image = channel_registrator.estimate(random_channel_misalignment, 0, 200, 3, 0.5, save_translation_masks=False,
                                                 translation_mask_save_path="", apply=True)
    aligned_image_2 = apply_channel_registration(random_channel_misalignment,
                                                 translation_masks=channel_registrator.translation_masks)

    assert (aligned_image==aligned_image_2).all()
