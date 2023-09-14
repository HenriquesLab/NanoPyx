from ..workflow import Workflow
from ..core.transform import  CRShiftAndMagnify, RadialGradientConvergence, GradientRobertsCross, eSRRF_ST
import numpy as np

# TODO check correlations and error map


def eSRRF(image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True, _force_run_type=None):
      """
      eSRRF analysis of an image
      """

      _eSRRF = Workflow((eSRRF_ST(), (image,), {'magnification': magnification, 'radius': radius, 'sensitivity': sensitivity, 'doIntensityWeighting': doIntensityWeighting}))

      return _eSRRF.run(_force_run_type=_force_run_type)
