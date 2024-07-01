from nanopyx.methods.workflow import Workflow
from nanopyx.core.transform import Radiality, CRShiftAndMagnify
import numpy as np


# Sample functions to use in the workflow
def test_workflow():
    image = np.random.random((100, 100))
    magnification = 2
    ringRadius = 1
    border = 0
    radialityPositivityConstraint = True
    doIntensityWeighting = True
    _SRRF = Workflow(
        (CRShiftAndMagnify(verbose=False), (image, 0, 0, magnification, magnification), {}),
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

    _SRRF.calculate()
