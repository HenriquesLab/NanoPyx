from .workflow import Workflow
from ..liquid import  CRShiftAndMagnify, RadialGradientConvergence, GradientRobertsCross
import numpy as np


def eSRRF(image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1,
          doIntensityWeighting: bool = True, interp_runtype1: str = "Threaded",
          interp_runtype2: str = "Threaded", interp_runtype3: str = "Threaded",
          gradrc_runtype: str = "Threaded", rgc_runtype: str = "Threaded"):
    """
    eSRRF analysis of an image
    """
    Gx_Gy_MAGNIFICATION = 2.0

    _eSRRF = Workflow((CRShiftAndMagnify(), (image, 0, 0, magnification, magnification), {"run_type": interp_runtype1}),
                    (GradientRobertsCross(), (image,), {"run_type": gradrc_runtype}),
                    (CRShiftAndMagnify(), ('PREV_RETURN_VALUE_1_0', 0, 0, magnification * Gx_Gy_MAGNIFICATION, magnification * Gx_Gy_MAGNIFICATION), {"run_type": interp_runtype2}),
                    (CRShiftAndMagnify(), ('PREV_RETURN_VALUE_1_1', 0, 0, magnification * Gx_Gy_MAGNIFICATION, magnification * Gx_Gy_MAGNIFICATION), {"run_type": interp_runtype3}),
                    (RadialGradientConvergence(), ('PREV_RETURN_VALUE_2_0', 'PREV_RETURN_VALUE_3_0', 'PREV_RETURN_VALUE_0_0'), {'magnification': magnification, 'radius': radius, 'sensitivity': sensitivity,'doIntensityWeighting': doIntensityWeighting, "run_type": rgc_runtype}))
    
    return _eSRRF