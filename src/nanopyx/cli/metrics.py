from __future__ import annotations

import argparse
import csv
import warnings

import numpy as np
from tifffile import imread, imwrite

from nanopyx.methods import calculate_decorr_analysis, calculate_error_map, calculate_frc


def _write_scalar_csv(path: str, metric_name: str, value: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow([metric_name, float(value)])


def _as_2d_image(image: np.ndarray, argument_name: str, plane_index: int = 0) -> np.ndarray:
    squeezed = np.squeeze(image)
    if squeezed.ndim == 2:
        return squeezed

    if squeezed.ndim == 3 and squeezed.shape[-1] in (3, 4):
        warnings.warn(
            f"{argument_name} has shape {image.shape}; converting RGB/RGBA input to a 2D grayscale image.",
            stacklevel=2,
        )
        return squeezed[..., :3].mean(axis=-1)

    if squeezed.ndim >= 3:
        planes = squeezed.reshape((-1,) + squeezed.shape[-2:])
        if plane_index < 0 or plane_index >= planes.shape[0]:
            raise ValueError(
                f"{argument_name} plane_index {plane_index} is out of range for image shape {image.shape}; "
                f"valid range is 0 to {planes.shape[0] - 1}."
            )
        warnings.warn(
            f"{argument_name} has shape {image.shape}; using plane_index={plane_index} for 2D error map calculation.",
            stacklevel=2,
        )
        return planes[plane_index]

    raise ValueError(f"{argument_name} must be a 2D image. Got shape {image.shape}.")


def linkinpy_calculate_error_map(
    input_image_reference: str,
    input_image_super_resolution: str,
    output_image: str,
    output_table_metrics: str,
    plane_index: int = 0,
) -> None:
    img_ref = _as_2d_image(imread(input_image_reference), "input_image_reference", plane_index)
    img_sr = _as_2d_image(imread(input_image_super_resolution), "input_image_super_resolution", plane_index)
    error_map, rse, rsp = calculate_error_map(img_ref=img_ref, img_sr=img_sr)
    imwrite(output_image, error_map)

    with open(output_table_metrics, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow(["RSE", float(rse)])
        writer.writerow(["RSP", float(rsp)])


def linkinpy_calculate_error_map_main() -> None:
    parser = argparse.ArgumentParser(description="Calculate SQUIRREL error map.")
    parser.add_argument("input_image_reference", help="Reference image path.")
    parser.add_argument("input_image_super_resolution", help="Super-resolution image path.")
    parser.add_argument("output_image", help="Output error map image path.")
    parser.add_argument("output_table_metrics", help="Output CSV path for RSE/RSP.")
    parser.add_argument("--plane-index", "--plane_index", dest="plane_index", type=int, default=0)
    args = parser.parse_args()
    linkinpy_calculate_error_map(
        input_image_reference=args.input_image_reference,
        input_image_super_resolution=args.input_image_super_resolution,
        output_image=args.output_image,
        output_table_metrics=args.output_table_metrics,
        plane_index=args.plane_index,
    )


def linkinpy_calculate_frc(
    input_image_frame_1: str,
    input_image_frame_2: str,
    output_table_frc: str,
    pixel_size: float = 1.0,
    units: str = "pixel",
    plot_frc_curve: bool = False,
) -> None:
    frame_1 = imread(input_image_frame_1)
    frame_2 = imread(input_image_frame_2)
    resolution = calculate_frc(
        frame_1=frame_1,
        frame_2=frame_2,
        pixel_size=pixel_size,
        units=units,
        plot_frc_curve=plot_frc_curve,
    )
    _write_scalar_csv(output_table_frc, "frc_resolution", float(resolution))


def linkinpy_calculate_frc_main() -> None:
    parser = argparse.ArgumentParser(description="Calculate FRC resolution.")
    parser.add_argument("input_image_frame_1", help="First image path.")
    parser.add_argument("input_image_frame_2", help="Second image path.")
    parser.add_argument("output_table_frc", help="Output CSV path.")
    parser.add_argument("--pixel-size", "--pixel_size", dest="pixel_size", type=float, default=1.0)
    parser.add_argument("--units", default="pixel")
    parser.add_argument("--plot-frc-curve", "--plot_frc_curve", dest="plot_frc_curve", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    linkinpy_calculate_frc(
        input_image_frame_1=args.input_image_frame_1,
        input_image_frame_2=args.input_image_frame_2,
        output_table_frc=args.output_table_frc,
        pixel_size=args.pixel_size,
        units=args.units,
        plot_frc_curve=args.plot_frc_curve,
    )


def linkinpy_calculate_decorr_analysis(
    input_image: str,
    output_table_decorr: str,
    rmin: float = 0.0,
    rmax: float = 1.0,
    n_r: int = 50,
    n_g: int = 10,
    pixel_size: float = 1.0,
    units: str = "pixel",
    roi: str = "0,0,0,0",
    plot_decorr_analysis: bool = False,
) -> None:
    roi_tuple = tuple(int(value.strip()) for value in roi.split(","))
    frame = imread(input_image)
    resolution = calculate_decorr_analysis(
        frame=frame,
        rmin=rmin,
        rmax=rmax,
        n_r=n_r,
        n_g=n_g,
        pixel_size=pixel_size,
        units=units,
        roi=roi_tuple,
        plot_decorr_analysis=plot_decorr_analysis,
    )
    _write_scalar_csv(output_table_decorr, "decorr_resolution", float(resolution))


def linkinpy_calculate_decorr_analysis_main() -> None:
    parser = argparse.ArgumentParser(description="Calculate decorrelation-based resolution.")
    parser.add_argument("input_image", help="Input image path.")
    parser.add_argument("output_table_decorr", help="Output CSV path.")
    parser.add_argument("--rmin", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=1.0)
    parser.add_argument("--n-r", "--n_r", dest="n_r", type=int, default=50)
    parser.add_argument("--n-g", "--n_g", dest="n_g", type=int, default=10)
    parser.add_argument("--pixel-size", "--pixel_size", dest="pixel_size", type=float, default=1.0)
    parser.add_argument("--units", default="pixel")
    parser.add_argument("--roi", default="0,0,0,0")
    parser.add_argument("--plot-decorr-analysis", "--plot_decorr_analysis", dest="plot_decorr_analysis", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    linkinpy_calculate_decorr_analysis(
        input_image=args.input_image,
        output_table_decorr=args.output_table_decorr,
        rmin=args.rmin,
        rmax=args.rmax,
        n_r=args.n_r,
        n_g=args.n_g,
        pixel_size=args.pixel_size,
        units=args.units,
        roi=args.roi,
        plot_decorr_analysis=args.plot_decorr_analysis,
    )


def linkinpy_Metrics_CalculateErrorMap(
    input_image_reference: str,
    input_image_super_resolution: str,
    output_image: str,
    output_table_metrics: str,
    plane_index: int = 0,
) -> None:
    linkinpy_calculate_error_map(
        input_image_reference=input_image_reference,
        input_image_super_resolution=input_image_super_resolution,
        output_image=output_image,
        output_table_metrics=output_table_metrics,
        plane_index=plane_index,
    )


def linkinpy_Metrics_CalculateFRC(
    input_image_frame_1: str,
    input_image_frame_2: str,
    output_table_frc: str,
    pixel_size: float = 1.0,
    units: str = "pixel",
    plot_frc_curve: bool = False,
) -> None:
    linkinpy_calculate_frc(
        input_image_frame_1=input_image_frame_1,
        input_image_frame_2=input_image_frame_2,
        output_table_frc=output_table_frc,
        pixel_size=pixel_size,
        units=units,
        plot_frc_curve=plot_frc_curve,
    )


def linkinpy_Metrics_ImageDecorrelationAnalysis(
    input_image: str,
    output_table_decorr: str,
    rmin: float = 0.0,
    rmax: float = 1.0,
    n_r: int = 50,
    n_g: int = 10,
    pixel_size: float = 1.0,
    units: str = "pixel",
    roi: str = "0,0,0,0",
    plot_decorr_analysis: bool = False,
) -> None:
    linkinpy_calculate_decorr_analysis(
        input_image=input_image,
        output_table_decorr=output_table_decorr,
        rmin=rmin,
        rmax=rmax,
        n_r=n_r,
        n_g=n_g,
        pixel_size=pixel_size,
        units=units,
        roi=roi,
        plot_decorr_analysis=plot_decorr_analysis,
    )
