from ..workflow import Workflow
from ...core.transform import eSRRF_ST
import numpy as np

# TODO check correlations and error map


def eSRRF(
    image,
    magnification: int = 5,
    radius: float = 1.5,
    sensitivity: float = 1,
    doIntensityWeighting: bool = True,
    _force_run_type=None,
):
    """
    Perform eSRRF analysis on an image.

    Args:
          image (numpy.ndarray): The input image for eSRRF analysis.
          magnification (int, optional): Magnification factor (default is 5).
          radius (float, optional): Radius parameter for eSRRF analysis (default is 1.5).
          sensitivity (float, optional): Sensitivity parameter for eSRRF analysis (default is 1).
          doIntensityWeighting (bool, optional): Enable intensity weighting (default is True).
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

    _eSRRF = Workflow(
        (
            eSRRF_ST(verbose=False),
            (image,),
            {
                "magnification": magnification,
                "radius": radius,
                "sensitivity": sensitivity,
                "doIntensityWeighting": doIntensityWeighting,
            },
        )
    )

    return _eSRRF.calculate(_force_run_type=_force_run_type)
