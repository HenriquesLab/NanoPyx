from __future__ import annotations

import argparse

from tifffile import imread, imwrite

from nanopyx.methods import non_local_means_denoising


def linkinpy_non_local_means_denoising(
    input_image: str,
    output_image: str,
    patch_size: int = 7,
    patch_distance: int = 11,
    h: float = 0.1,
    sigma: float = 0.0,
) -> None:
    image = imread(input_image)
    output = non_local_means_denoising(
        image,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=h,
        sigma=sigma,
    )
    imwrite(output_image, output)


def linkinpy_non_local_means_denoising_main() -> None:
    parser = argparse.ArgumentParser(description="Run NanoPyx non-local means denoising.")
    parser.add_argument("input_image", help="Input image path.")
    parser.add_argument("output_image", help="Output image path.")
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--patch_distance", type=int, default=11)
    parser.add_argument("--h", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.0)
    args = parser.parse_args()
    linkinpy_non_local_means_denoising(
        input_image=args.input_image,
        output_image=args.output_image,
        patch_size=args.patch_size,
        patch_distance=args.patch_distance,
        h=args.h,
        sigma=args.sigma,
    )
