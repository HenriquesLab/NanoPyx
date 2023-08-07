from .workflow import Workflow
from ..liquid import  CRShiftAndMagnify, RadialGradientConvergence, GradientRobertsCross, eSRRF_ST
import numpy as np


def eSRRF(image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True,
          interp_runtype1="Threaded", interp_runtype2="Threaded", interp_runtype3="Threaded", gradrc_runtype="Threaded", rgc_runtype="Threaded"):
      """
      eSRRF analysis of an image
      """
      Gx_Gy_MAGNIFICATION = 2.0
      
      if doIntensityWeighting:

            _eSRRF = Workflow((CRShiftAndMagnify(), (image, 0, 0, magnification, magnification), {'run_type':interp_runtype1}),
                              (GradientRobertsCross(), (image,), {'run_type':gradrc_runtype}),
                              (CRShiftAndMagnify(), ('PREV_RETURN_VALUE_1_0', 0, 0, magnification * Gx_Gy_MAGNIFICATION, magnification * Gx_Gy_MAGNIFICATION), {'run_type':interp_runtype2}),
                              (CRShiftAndMagnify(), ('PREV_RETURN_VALUE_1_1', 0, 0, magnification * Gx_Gy_MAGNIFICATION, magnification * Gx_Gy_MAGNIFICATION), {'run_type':interp_runtype3}),
                              (RadialGradientConvergence(), ('PREV_RETURN_VALUE_2_0', 'PREV_RETURN_VALUE_3_0', 'PREV_RETURN_VALUE_0_0'), {'magnification': magnification, 'radius': radius, 'sensitivity': sensitivity,'doIntensityWeighting': doIntensityWeighting,'run_type':rgc_runtype}))
      else:
            _eSRRF = Workflow((GradientRobertsCross(), (image,), {'run_type':gradrc_runtype}),
                              (CRShiftAndMagnify(), ('PREV_RETURN_VALUE_1_0', 0, 0, magnification * Gx_Gy_MAGNIFICATION, magnification * Gx_Gy_MAGNIFICATION), {'run_type':interp_runtype2}),
                              (CRShiftAndMagnify(), ('PREV_RETURN_VALUE_1_1', 0, 0, magnification * Gx_Gy_MAGNIFICATION, magnification * Gx_Gy_MAGNIFICATION), {'run_type':interp_runtype3}),
                              (RadialGradientConvergence(), ('PREV_RETURN_VALUE_2_0', 'PREV_RETURN_VALUE_3_0', image), {'magnification': magnification, 'radius': radius, 'sensitivity': sensitivity,'doIntensityWeighting': doIntensityWeighting,'run_type':rgc_runtype}))

      return _eSRRF

def eSRRF_2(image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True,
          interp_runtype1="Threaded", interp_runtype2="Threaded", interp_runtype3="Threaded", gradrc_runtype="Threaded", rgc_runtype="Threaded"):
      """
      eSRRF analysis of an image
      """
      Gx_Gy_MAGNIFICATION = 2.0

      _eSRRF = Workflow((CRShiftAndMagnify(), (image, 0, 0, magnification, magnification), {'run_type':interp_runtype1}),
                        (CRShiftAndMagnify(), (image, 0, 0, magnification * Gx_Gy_MAGNIFICATION, magnification * Gx_Gy_MAGNIFICATION), {'run_type':interp_runtype2}),
                        (GradientRobertsCross(),( "PREV_RETURN_VALUE_1_0",), {'run_type':gradrc_runtype}),
                        (RadialGradientConvergence(), ('PREV_RETURN_VALUE_2_0', 'PREV_RETURN_VALUE_2_1', 'PREV_RETURN_VALUE_0_0'), {'magnification': magnification, 'radius': radius, 'sensitivity': sensitivity,'doIntensityWeighting': doIntensityWeighting,'run_type':rgc_runtype}))

      return _eSRRF

def eSRRF_3(image, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1, doIntensityWeighting: bool = True):
      """
      eSRRF analysis of an image
      """

      _eSRRF = Workflow((eSRRF_ST(), (image,), {'magnification': magnification, 'radius': radius, 'sensitivity': sensitivity, 'doIntensityWeighting': doIntensityWeighting}))

      return _eSRRF
