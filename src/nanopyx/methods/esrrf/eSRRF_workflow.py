from ..workflow import Workflow
from ...core.transform import eSRRF_ST
from ...core.transform.mpcorrector import macro_pixel_corrector
from ...core.transform.sr_temporal_correlations import (
    calculate_eSRRF_temporal_correlations,
)
import numpy as np

# TODO check correlations and error map


def eSRRF(
    image,
    magnification: int = 5,
    grad_magnification: int = 2,
    radius: float = 1.5,
    sensitivity: float = 1,
    frames_per_timepoint: int = 0,
    temporal_correlation: str = "AVG",
    doIntensityWeighting: bool = True,
    macro_pixel_correction: bool = True,
    _force_run_type=None,
):
    """
    Perform eSRRF analysis on an image.

    Args:
          image (numpy.ndarray): The input image for eSRRF analysis.
          magnification (int, optional): Magnification factor (default is 5).
          radius (float, optional): Radius parameter for eSRRF analysis (default is 1.5).
          sensitivity (float, optional): Sensitivity parameter for eSRRF analysis (default is 1).
          frames_per_timepoint (int, optional): Number of frames per timepoint (default is 0, which means all frames are used).
          temporal_correlation (str, optional): Type of temporal correlation to calculate. Options are: AVG, VAR or TAC2 (default is "AVG").
          doIntensityWeighting (bool, optional): Enable intensity weighting (default is True).
          macro_pixel_correction (bool, optional): Enable macro pixel correction (default is True).
          _force_run_type (str, optional): Force a specific run type for the analysis (default is None).

    Returns:
          numpy.ndarray: The result of eSRRF analysis, typically representing the localizations.

    Example:
          result = eSRRF(image, magnification=5, radius=1.5, sensitivity=1, doIntensityWeighting=True)

    Note:
          - eSRRF (enhanced Super-Resolution Radial Fluctuations) is a method for super-resolution localization microscopy.
          - This function sets up a workflow to perform eSRRF analysis on the input image.
          - The workflow includes eSRRF_ST as a step and can be customized with various parameters.
          - The result is typically a numpy array representing the localized points.

    See Also:
          - eSRRF_ST: The eSRRF step that performs the actual analysis.
          - Workflow: The class used to define and run analysis workflows.
    """

    if frames_per_timepoint == 0:
        frames_per_timepoint = image.shape[0]
    elif frames_per_timepoint > image.shape[0]:
        frames_per_timepoint = image.shape[0]

    number_of_timepoints = image.shape[0] // frames_per_timepoint
    if image.shape[0] % frames_per_timepoint != 0:
        number_of_timepoints += 1

    output_array = np.zeros(
        (
            number_of_timepoints,
            image.shape[1] * magnification,
            image.shape[2] * magnification,
        ),
        dtype=np.float32,
    )

    for i in range(number_of_timepoints):

        _eSRRF = Workflow(
            (
                eSRRF_ST(verbose=False),
                (
                    image[
                        frames_per_timepoint
                        * i : frames_per_timepoint
                        * (i + 1)
                    ],
                ),
                {
                    "magnification": magnification,
                    "grad_magnification": grad_magnification,
                    "radius": radius,
                    "sensitivity": sensitivity,
                    "doIntensityWeighting": doIntensityWeighting,
                },
            )
        )
        if macro_pixel_correction:
            output_array[i] = macro_pixel_corrector(
                np.expand_dims(
                    np.asarray(
                        calculate_eSRRF_temporal_correlations(
                            _eSRRF.calculate(_force_run_type=_force_run_type)[
                                0
                            ],
                            temporal_correlation,
                        )
                    ),
                    axis=0,
                ),
                magnification=magnification,
            )
        else:
            output_array[i] = np.asarray(
                calculate_eSRRF_temporal_correlations(
                    _eSRRF.calculate(_force_run_type=_force_run_type)[0],
                    temporal_correlation,
                )
            )

    return np.squeeze(output_array.astype(np.float32))
