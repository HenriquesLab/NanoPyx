from .workflow import Workflow
from ..liquid import Radiality, CRShiftAndMagnify


import numpy as np


def SRRF(image, magnification, ringRadius, border, radialityPositivityConstraint, doIntensityWeighting):
    """
    SRRF analysis of a single image
    """



    _SRRF = Workflow((CRShiftAndMagnify(),(image, 0,0,magnification,magnification),{}),
                    (Radiality(),(image, 'PREV_RETURN_VALUE_0_0'),{'magnification':magnification, 'ringRadius':ringRadius, 'border':border,'radialityPositivityConstraint':radialityPositivityConstraint,'doIntensityWeighting':doIntensityWeighting}))
    
    
    return _SRRF