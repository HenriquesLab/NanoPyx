from ..workflow import Workflow
from ...core.transform import Radiality, CRShiftAndMagnify
from ...core.transform.mpcorrector import macro_pixel_corrector


import numpy as np


def SRRF(
    image,
    magnification=5,
    ringRadius=0.5,
    border=0,
    radialityPositivityConstraint=True,
    doIntensityWeighting=True,
    macro_pixel_correction=True,
    _force_run_type=None,
):
    """
    Perform SRRF (Super-Resolution Radial Fluctuations) analysis on a single image.

    Args:
        image (numpy.ndarray): The input image for SRRF analysis.
        magnification (int, optional): Magnification factor (default is 5).
        ringRadius (float, optional): Radius of the ring for radiality analysis (default is 0.5).
        border (int, optional): Border parameter for radiality analysis (default is 0).
        radialityPositivityConstraint (bool, optional): Enable radiality positivity constraint (default is True).
        doIntensityWeighting (bool, optional): Enable intensity weighting (default is True).
        macro_pixel_correction (bool, optional): Enable macro pixel correction (default is True).
        _force_run_type (str, optional): Force a specific run type for the analysis (default is None).

    Returns:
        numpy.ndarray: The result of SRRF analysis, typically representing super-resolved structures.

    Example:
        result = SRRF(image, magnification=5, ringRadius=0.5, border=0, radialityPositivityConstraint=True, doIntensityWeighting=True)

    Note:
        - SRRF (Super-Resolution Radial Fluctuations) is a method for super-resolution microscopy.
        - This function sets up a workflow to perform SRRF analysis on the input image.
        - The workflow includes CRShiftAndMagnify and Radiality as steps and can be customized with various parameters.
        - The result is typically a numpy array representing super-resolved structures.

    See Also:
        - CRShiftAndMagnify: A step that performs coordinate transformation and magnification.
        - Radiality: A step that calculates radiality for super-resolution analysis.
        - Workflow: The class used to define and run analysis workflows.
    """

    _SRRF = Workflow(
        (
            CRShiftAndMagnify(verbose=False),
            (image, 0, 0, magnification, magnification),
            {},
        ),
        (
            Radiality(verbose=False),
            (image, "PREV_RETURN_VALUE_0_0"),
            {
                "magnification": magnification,
                "ringRadius": ringRadius,
                "border": border,
                "radialityPositivityConstraint": radialityPositivityConstraint,
                "doIntensityWeighting": doIntensityWeighting,
            },
        ),
    )

    if macro_pixel_correction:
        return macro_pixel_corrector(
            _SRRF.calculate(_force_run_type=_force_run_type)[0],
            magnification=magnification,
        )

    else:
        return _SRRF.calculate(_force_run_type=_force_run_type)[0]
