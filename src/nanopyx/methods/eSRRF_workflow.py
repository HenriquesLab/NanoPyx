from .workflow import Workflow
from ..liquid import  CRShiftAndMagnify, RadialGradientConvergence, GradientRobertsCross
import numpy as np


def eSRRF(image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True):
    """
    eSRRF analysis of an image
    """
    Gx_Gy_MAGNIFICATION = 2.0

    _eSRRF = Workflow((CRShiftAndMagnify(), (image, 0, 0, magnification, magnification), {}),
                    (GradientRobertsCross(), (image), {}),
                    (CRShiftAndMagnify(), ('PREV_RETURN_VALUE_1', 0, 0, magnification * Gx_Gy_MAGNIFICATION, magnification * Gx_Gy_MAGNIFICATION), {}),
                    (RadialGradientConvergence(),('PREV_RETURN_VALUE_0','PREV_RETURN_VALUE_2'),{'magnification': magnification, 'radius': radius, 'sensitivity': sensitivity,'doIntensityWeighting': doIntensityWeighting}))
    
    return _eSRRF