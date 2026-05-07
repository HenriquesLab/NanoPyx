from __future__ import annotations

import argparse

from tifffile import imread, imwrite

from nanopyx.methods import (
    apply_channel_registration,
    apply_drift_alignment,
    estimate_channel_registration,
    estimate_drift_alignment,
)


def linkinpy_estimate_channel_registration(
    input_image: str,
    output_image: str,
    ref_channel: int,
    max_shift: float,
    blocks_per_axis: int,
    min_similarity: float,
    save_translation_masks: bool = True,
    translation_mask_save_path: str = None,
    apply: bool = True,
) -> None:
    image = imread(input_image)
    output = estimate_channel_registration(
        image_array=image,
        ref_channel=ref_channel,
        max_shift=max_shift,
        blocks_per_axis=blocks_per_axis,
        min_similarity=min_similarity,
        save_translation_masks=save_translation_masks,
        translation_mask_save_path=translation_mask_save_path,
        apply=apply,
    )
    if output is not None:
        imwrite(output_image, output)


def linkinpy_estimate_channel_registration_main() -> None:
    parser = argparse.ArgumentParser(description="Estimate and optionally apply channel registration.")
    parser.add_argument("input_image", help="Input multi-channel image path.")
    parser.add_argument("output_image", help="Output aligned image path.")
    parser.add_argument("--ref_channel", type=int, required=True)
    parser.add_argument("--max_shift", type=float, required=True)
    parser.add_argument("--blocks_per_axis", type=int, required=True)
    parser.add_argument("--min_similarity", type=float, required=True)
    parser.add_argument("--save_translation_masks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--translation_mask_save_path", default=None)
    parser.add_argument("--apply", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    linkinpy_estimate_channel_registration(
        input_image=args.input_image,
        output_image=args.output_image,
        ref_channel=args.ref_channel,
        max_shift=args.max_shift,
        blocks_per_axis=args.blocks_per_axis,
        min_similarity=args.min_similarity,
        save_translation_masks=args.save_translation_masks,
        translation_mask_save_path=args.translation_mask_save_path,
        apply=args.apply,
    )


def linkinpy_apply_channel_registration(
    input_image: str,
    output_image: str,
    input_translation_masks: str,
) -> None:
    image = imread(input_image)
    translation_masks = imread(input_translation_masks)
    output = apply_channel_registration(image_array=image, translation_masks=translation_masks)
    imwrite(output_image, output)


def linkinpy_apply_channel_registration_main() -> None:
    parser = argparse.ArgumentParser(description="Apply channel registration using translation masks.")
    parser.add_argument("input_image", help="Input multi-channel image path.")
    parser.add_argument("output_image", help="Output aligned image path.")
    parser.add_argument("--input_translation_masks", required=True)
    args = parser.parse_args()
    linkinpy_apply_channel_registration(
        input_image=args.input_image,
        output_image=args.output_image,
        input_translation_masks=args.input_translation_masks,
    )


def linkinpy_estimate_drift_alignment(
    input_image: str,
    output_image: str,
    save_as_npy: bool = True,
    save_drift_table_path: str = None,
    time_averaging: int = 1,
    max_expected_drift: float = 6.0,
    ref_option: int = 0,
    apply: bool = True,
) -> None:
    image = imread(input_image)
    output = estimate_drift_alignment(
        image_array=image,
        save_as_npy=save_as_npy,
        save_drift_table_path=save_drift_table_path,
        time_averaging=time_averaging,
        max_expected_drift=max_expected_drift,
        ref_option=ref_option,
        apply=apply,
    )
    if output is not None:
        imwrite(output_image, output)


def linkinpy_estimate_drift_alignment_main() -> None:
    parser = argparse.ArgumentParser(description="Estimate and optionally apply drift alignment.")
    parser.add_argument("input_image", help="Input timelapse image path.")
    parser.add_argument("output_image", help="Output drift-corrected image path.")
    parser.add_argument("--save_as_npy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_drift_table_path", default=None)
    parser.add_argument("--time_averaging", type=int, default=1)
    parser.add_argument("--max_expected_drift", type=float, default=6)
    parser.add_argument("--ref_option", type=int, default=0)
    parser.add_argument("--apply", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    linkinpy_estimate_drift_alignment(
        input_image=args.input_image,
        output_image=args.output_image,
        save_as_npy=args.save_as_npy,
        save_drift_table_path=args.save_drift_table_path,
        time_averaging=args.time_averaging,
        max_expected_drift=args.max_expected_drift,
        ref_option=args.ref_option,
        apply=args.apply,
    )


def linkinpy_apply_drift_alignment(
    input_image: str,
    output_image: str,
    input_drift_table: str,
) -> None:
    image = imread(input_image)
    output = apply_drift_alignment(image_array=image, path=input_drift_table)
    imwrite(output_image, output)


def linkinpy_apply_drift_alignment_main() -> None:
    parser = argparse.ArgumentParser(description="Apply drift alignment using a drift table.")
    parser.add_argument("input_image", help="Input timelapse image path.")
    parser.add_argument("output_image", help="Output drift-corrected image path.")
    parser.add_argument("--input_drift_table", required=True)
    args = parser.parse_args()
    linkinpy_apply_drift_alignment(
        input_image=args.input_image,
        output_image=args.output_image,
        input_drift_table=args.input_drift_table,
    )


def linkinpy_Registration_EstimateChannelRegistration(
    input_image: str,
    output_image: str,
    ref_channel: int,
    max_shift: float,
    blocks_per_axis: int,
    min_similarity: float,
    save_translation_masks: bool = True,
    translation_mask_save_path: str = None,
    apply: bool = True,
) -> None:
    linkinpy_estimate_channel_registration(
        input_image=input_image,
        output_image=output_image,
        ref_channel=ref_channel,
        max_shift=max_shift,
        blocks_per_axis=blocks_per_axis,
        min_similarity=min_similarity,
        save_translation_masks=save_translation_masks,
        translation_mask_save_path=translation_mask_save_path,
        apply=apply,
    )


def linkinpy_Registration_ApplyChannelRegistration(
    input_image: str,
    output_image: str,
    input_translation_masks: str,
) -> None:
    linkinpy_apply_channel_registration(
        input_image=input_image,
        output_image=output_image,
        input_translation_masks=input_translation_masks,
    )


def linkinpy_Registration_EstimateDriftAlignment(
    input_image: str,
    output_image: str,
    save_as_npy: bool = True,
    save_drift_table_path: str = None,
    time_averaging: int = 1,
    max_expected_drift: float = 6.0,
    ref_option: int = 0,
    apply: bool = True,
) -> None:
    linkinpy_estimate_drift_alignment(
        input_image=input_image,
        output_image=output_image,
        save_as_npy=save_as_npy,
        save_drift_table_path=save_drift_table_path,
        time_averaging=time_averaging,
        max_expected_drift=max_expected_drift,
        ref_option=ref_option,
        apply=apply,
    )


def linkinpy_Registration_ApplyDriftAlignment(
    input_image: str,
    output_image: str,
    input_drift_table: str,
) -> None:
    linkinpy_apply_drift_alignment(
        input_image=input_image,
        output_image=output_image,
        input_drift_table=input_drift_table,
    )
