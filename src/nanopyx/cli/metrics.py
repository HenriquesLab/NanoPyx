from __future__ import annotations

import argparse
import csv

from tifffile import imread, imwrite

from nanopyx.methods import calculate_decorr_analysis, calculate_error_map, calculate_frc


def _write_scalar_csv(path: str, metric_name: str, value: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["metric", "value"])
        writer.writerow([metric_name, float(value)])


def linkinpy_calculate_error_map(
    input_reference: str,
    input_super_resolution: str,
    output_image: str,
    output_metrics: str = None,
) -> None:
    img_ref = imread(input_reference)
    img_sr = imread(input_super_resolution)
    error_map, rse, rsp = calculate_error_map(img_ref=img_ref, img_sr=img_sr)
    imwrite(output_image, error_map)

    if output_metrics:
        with open(output_metrics, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["metric", "value"])
            writer.writerow(["RSE", float(rse)])
            writer.writerow(["RSP", float(rsp)])


def linkinpy_calculate_error_map_main() -> None:
    parser = argparse.ArgumentParser(description="Calculate SQUIRREL error map.")
    parser.add_argument("input_reference", help="Reference image path.")
    parser.add_argument("input_super_resolution", help="Super-resolution image path.")
    parser.add_argument("output_image", help="Output error map image path.")
    parser.add_argument("--output_metrics", default=None, help="Optional output CSV path for RSE/RSP.")
    args = parser.parse_args()
    linkinpy_calculate_error_map(
        input_reference=args.input_reference,
        input_super_resolution=args.input_super_resolution,
        output_image=args.output_image,
        output_metrics=args.output_metrics,
    )


def linkinpy_calculate_frc(
    input_frame_1: str,
    input_frame_2: str,
    output_csv: str,
    pixel_size: float = 1.0,
    units: str = "pixel",
    plot_frc_curve: bool = False,
) -> None:
    frame_1 = imread(input_frame_1)
    frame_2 = imread(input_frame_2)
    resolution = calculate_frc(
        frame_1=frame_1,
        frame_2=frame_2,
        pixel_size=pixel_size,
        units=units,
        plot_frc_curve=plot_frc_curve,
    )
    _write_scalar_csv(output_csv, "frc_resolution", float(resolution))


def linkinpy_calculate_frc_main() -> None:
    parser = argparse.ArgumentParser(description="Calculate FRC resolution.")
    parser.add_argument("input_frame_1", help="First image path.")
    parser.add_argument("input_frame_2", help="Second image path.")
    parser.add_argument("output_csv", help="Output CSV path.")
    parser.add_argument("--pixel_size", type=float, default=1.0)
    parser.add_argument("--units", default="pixel")
    parser.add_argument("--plot_frc_curve", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    linkinpy_calculate_frc(
        input_frame_1=args.input_frame_1,
        input_frame_2=args.input_frame_2,
        output_csv=args.output_csv,
        pixel_size=args.pixel_size,
        units=args.units,
        plot_frc_curve=args.plot_frc_curve,
    )


def linkinpy_calculate_decorr_analysis(
    input_image: str,
    output_csv: str,
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
    _write_scalar_csv(output_csv, "decorr_resolution", float(resolution))


def linkinpy_calculate_decorr_analysis_main() -> None:
    parser = argparse.ArgumentParser(description="Calculate decorrelation-based resolution.")
    parser.add_argument("input_image", help="Input image path.")
    parser.add_argument("output_csv", help="Output CSV path.")
    parser.add_argument("--rmin", type=float, default=0.0)
    parser.add_argument("--rmax", type=float, default=1.0)
    parser.add_argument("--n_r", type=int, default=50)
    parser.add_argument("--n_g", type=int, default=10)
    parser.add_argument("--pixel_size", type=float, default=1.0)
    parser.add_argument("--units", default="pixel")
    parser.add_argument("--roi", default="0,0,0,0")
    parser.add_argument("--plot_decorr_analysis", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    linkinpy_calculate_decorr_analysis(
        input_image=args.input_image,
        output_csv=args.output_csv,
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
    input_reference: str,
    input_super_resolution: str,
    output_image: str,
    output_metrics: str = None,
) -> None:
    linkinpy_calculate_error_map(
        input_reference=input_reference,
        input_super_resolution=input_super_resolution,
        output_image=output_image,
        output_metrics=output_metrics,
    )


def linkinpy_Metrics_CalculateFRC(
    input_frame_1: str,
    input_frame_2: str,
    output_csv: str,
    pixel_size: float = 1.0,
    units: str = "pixel",
    plot_frc_curve: bool = False,
) -> None:
    linkinpy_calculate_frc(
        input_frame_1=input_frame_1,
        input_frame_2=input_frame_2,
        output_csv=output_csv,
        pixel_size=pixel_size,
        units=units,
        plot_frc_curve=plot_frc_curve,
    )


def linkinpy_Metrics_ImageDecorrelationAnalysis(
    input_image: str,
    output_csv: str,
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
        output_csv=output_csv,
        rmin=rmin,
        rmax=rmax,
        n_r=n_r,
        n_g=n_g,
        pixel_size=pixel_size,
        units=units,
        roi=roi,
        plot_decorr_analysis=plot_decorr_analysis,
    )
