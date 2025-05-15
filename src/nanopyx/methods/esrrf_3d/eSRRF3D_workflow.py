from ..workflow import Workflow
from ...core.transform._le_esrrf3d import eSRRF3D as eSRRF3D_ST
from ...core.transform.mpcorrector import macro_pixel_corrector

import numpy as np


def eSRRF3D(
    img,
    magnification_xy=2,
    magnification_z=2,
    radius: float = 1.5,
    radius_z: float = 0.5,
    voxel_ratio: float = 4.0,
    sensitivity: float = 1,
    mode: str = "average",
    doIntensityWeighting: bool = True,
    macro_pixel_correction: bool = False,
    _force_run_type=None,
):
    """
    #TODO
    """

    _eSRRF3D = Workflow(
        (
            eSRRF3D_ST(verbose=False),
            (img,),
            {
                "magnification_xy": magnification_xy,
                "magnification_z": magnification_z,
                "radius": radius,
                "radius_z": radius_z,
                "voxel_ratio": voxel_ratio,
                "sensitivity": sensitivity,
                "mode": mode,
                "doIntensityWeighting": doIntensityWeighting,
            },
        )
    )
    if macro_pixel_correction:
        return macro_pixel_corrector(
            _eSRRF3D.calculate(_force_run_type=_force_run_type)[0],
            magnification=magnification_xy,
        )
    else:
        return _eSRRF3D.calculate(_force_run_type=_force_run_type)[0]
