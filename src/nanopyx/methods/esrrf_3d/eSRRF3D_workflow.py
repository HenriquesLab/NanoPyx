from ..workflow import Workflow
from ...core.transform._le_esrrf3d import eSRRF3D as eSRRF3D_ST
from ...core.transform.mpcorrector import macro_pixel_corrector

import numpy as np


def eSRRF3D(
    img,
    magnification_xy=2,
    magnification_z=2,
    radius: float = 1.5,
    PSF_ratio: float = 2.8,
    voxel_ratio: float = 4.0,
    sensitivity: float = 1,
    frames_per_timepoint: int = 0,
    mode: str = "average",
    doIntensityWeighting: bool = True,
    macro_pixel_correction: bool = True,
    _force_run_type=None,
):
    """
    Perform eSRRF3D analysis on an image.

    Args:
        img (numpy.ndarray): The input image for eSRRF3D analysis.
        magnification_xy (int, optional): Magnification factor in XY plane (default is 2).
        magnification_z (int, optional): Magnification factor in Z plane (default is 2).
        radius (float, optional): Radius parameter for eSRRF3D analysis (default is 1.5).
        PSF_ratio (float, optional): Ratio of PSF shape as Z/XY for eSRRF3D analysis (default is 2.8).
        voxel_ratio (float, optional): Ratio of voxel size in XY to Z direction (default is 4.0).
        frames_per_timepoint (int, optional): Number of frames per timepoint (default is 0, which means all frames are used).
        sensitivity (float, optional): Sensitivity parameter for eSRRF3D analysis (default is 1).
        mode (str, optional): Time projection mode (default is "average").
        doIntensityWeighting (bool, optional): Enable intensity weighting (default is True).
        macro_pixel_correction (bool, optional): Enable macro pixel correction (default is True).
        _force_run_type (str, optional): Force a specific run type for the analysis (default is None).

    Returns:
        numpy.ndarray: The result of eSRRF3D analysis, typically representing the localizations.

    Example:
        result = eSRRF3D(image, magnification_xy=2, magnification_z=2, radius=1.5, sensitivity=1, doIntensityWeighting=True)

    Note:
        - eSRRF3D (enhanced Super-Resolution Radial Fluctuations 3D) is a method for super-resolution localization microscopy in three dimensions.
        - This function sets up a workflow to perform eSRRF3D analysis on the input image.

    See Also:
        - eSRRF3D_ST: The eSRRF3D step that performs the actual analysis.
        - Workflow: The class used to define and run analysis workflows.

    """

    if frames_per_timepoint == 0:
        frames_per_timepoint = img.shape[0]
    elif frames_per_timepoint > img.shape[0]:
        frames_per_timepoint = img.shape[0]

    number_of_timepoints = img.shape[0] // frames_per_timepoint
    if img.shape[0] % frames_per_timepoint != 0:
        number_of_timepoints += 1

    output_array = np.zeros(
        (
            number_of_timepoints,
            img.shape[1] * magnification_z,
            img.shape[2] * magnification_xy,
            img.shape[3] * magnification_xy,
        ),
        dtype=np.float32,
    )

    for i in range(number_of_timepoints):
        _eSRRF3D = Workflow(
            (
                eSRRF3D_ST(verbose=False),
                (
                    img[
                        frames_per_timepoint
                        * i : frames_per_timepoint
                        * (i + 1)
                    ],
                ),
                {
                    "magnification_xy": magnification_xy,
                    "magnification_z": magnification_z,
                    "radius": radius,
                    "PSF_ratio": PSF_ratio,
                    "voxel_ratio": voxel_ratio,
                    "sensitivity": sensitivity,
                    "mode": mode,
                    "doIntensityWeighting": doIntensityWeighting,
                },
            )
        )
        if macro_pixel_correction:
            output_array[i] = macro_pixel_corrector(
                _eSRRF3D.calculate(_force_run_type=_force_run_type)[0],
                magnification=magnification_xy,
            )
        else:
            output_array[i] = _eSRRF3D.calculate(
                _force_run_type=_force_run_type
            )[0]

    return output_array.astype(np.float32)
