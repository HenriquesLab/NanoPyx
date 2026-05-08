from __future__ import annotations

import argparse

from tifffile import imread, imwrite

from nanopyx.methods import SRRF, eSRRF


def linkinpy_SRRF(
    input_image: str,
    output_image: str,
    magnification: int = 5,
    ringRadius: float = 0.5,
    border: int = 0,
    radialityPositivityConstraint: bool = True,
    doIntensityWeighting: bool = True,
    macro_pixel_correction: bool = True,
) -> None:
    image = imread(input_image)
    output = SRRF(
        image=image,
        magnification=magnification,
        ringRadius=ringRadius,
        border=border,
        radialityPositivityConstraint=radialityPositivityConstraint,
        doIntensityWeighting=doIntensityWeighting,
        macro_pixel_correction=macro_pixel_correction,
    )
    imwrite(output_image, output)


def linkinpy_SRRF_main() -> None:
    parser = argparse.ArgumentParser(description="Run NanoPyx SRRF.")
    parser.add_argument("input_image", help="Input image path.")
    parser.add_argument("output_image", help="Output image path.")
    parser.add_argument("--magnification", type=int, default=5)
    parser.add_argument("--ring-radius", "--ringRadius", dest="ringRadius", type=float, default=0.5)
    parser.add_argument("--border", type=int, default=0)
    parser.add_argument(
        "--radiality-positivity-constraint",
        "--radialityPositivityConstraint",
        dest="radialityPositivityConstraint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--intensity-weighting",
        "--doIntensityWeighting",
        dest="doIntensityWeighting",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--macro-pixel-correction",
        "--macro_pixel_correction",
        dest="macro_pixel_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    linkinpy_SRRF(
        input_image=args.input_image,
        output_image=args.output_image,
        magnification=args.magnification,
        ringRadius=args.ringRadius,
        border=args.border,
        radialityPositivityConstraint=args.radialityPositivityConstraint,
        doIntensityWeighting=args.doIntensityWeighting,
        macro_pixel_correction=args.macro_pixel_correction,
    )


def linkinpy_eSRRF(
    input_image: str,
    output_image: str,
    magnification: int = 5,
    radius: float = 1.5,
    sensitivity: float = 1.0,
    frames_per_timepoint: int = 0,
    temporal_correlation: str = "AVG",
    use_fht: bool = False,
    doIntensityWeighting: bool = True,
    macro_pixel_correction: bool = True,
    pad_edges: bool = False,
) -> None:
    image = imread(input_image)
    output = eSRRF(
        image=image,
        magnification=magnification,
        radius=radius,
        sensitivity=sensitivity,
        frames_per_timepoint=frames_per_timepoint,
        temporal_correlation=temporal_correlation,
        use_fht=use_fht,
        doIntensityWeighting=doIntensityWeighting,
        macro_pixel_correction=macro_pixel_correction,
        pad_edges=pad_edges,
    )
    imwrite(output_image, output)


def linkinpy_eSRRF_main() -> None:
    parser = argparse.ArgumentParser(description="Run NanoPyx eSRRF.")
    parser.add_argument("input_image", help="Input image path.")
    parser.add_argument("output_image", help="Output image path.")
    parser.add_argument("--magnification", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--frames-per-timepoint", "--frames_per_timepoint", dest="frames_per_timepoint", type=int, default=0)
    parser.add_argument("--temporal-correlation", "--temporal_correlation", dest="temporal_correlation", default="AVG")
    parser.add_argument(
        "--use-fht", "--use_fht", dest="use_fht", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--intensity-weighting",
        "--doIntensityWeighting",
        dest="doIntensityWeighting",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--macro-pixel-correction",
        "--macro_pixel_correction",
        dest="macro_pixel_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pad-edges", "--pad_edges", dest="pad_edges", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    linkinpy_eSRRF(
        input_image=args.input_image,
        output_image=args.output_image,
        magnification=args.magnification,
        radius=args.radius,
        sensitivity=args.sensitivity,
        frames_per_timepoint=args.frames_per_timepoint,
        temporal_correlation=args.temporal_correlation,
        use_fht=args.use_fht,
        doIntensityWeighting=args.doIntensityWeighting,
        macro_pixel_correction=args.macro_pixel_correction,
        pad_edges=args.pad_edges,
    )


def linkinpy_SuperResolution_SRRF(
    input_image: str,
    output_image: str,
    magnification: int = 5,
    ringRadius: float = 0.5,
    border: int = 0,
    radialityPositivityConstraint: bool = True,
    doIntensityWeighting: bool = True,
    macro_pixel_correction: bool = True,
) -> None:
    linkinpy_SRRF(
        input_image=input_image,
        output_image=output_image,
        magnification=magnification,
        ringRadius=ringRadius,
        border=border,
        radialityPositivityConstraint=radialityPositivityConstraint,
        doIntensityWeighting=doIntensityWeighting,
        macro_pixel_correction=macro_pixel_correction,
    )


def linkinpy_SuperResolution_eSRRF(
    input_image: str,
    output_image: str,
    magnification: int = 5,
    radius: float = 1.5,
    sensitivity: float = 1.0,
    frames_per_timepoint: int = 0,
    temporal_correlation: str = "AVG",
    use_fht: bool = False,
    doIntensityWeighting: bool = True,
    macro_pixel_correction: bool = True,
    pad_edges: bool = False,
) -> None:
    linkinpy_eSRRF(
        input_image=input_image,
        output_image=output_image,
        magnification=magnification,
        radius=radius,
        sensitivity=sensitivity,
        frames_per_timepoint=frames_per_timepoint,
        temporal_correlation=temporal_correlation,
        use_fht=use_fht,
        doIntensityWeighting=doIntensityWeighting,
        macro_pixel_correction=macro_pixel_correction,
        pad_edges=pad_edges,
    )
