from ...core.transform._le_esrrf3d import eSRRF3D
from ...core.transform.sr_temporal_correlations import (
    calculate_eSRRF3d_temporal_correlations,
)


def run_esrrf3d(
    img,
    mode="average",
    magnification_xy=2,
    magnification_z=2,
    radius=1.5,
    sensitivity=1,
    voxel_ratio=4,
    doIntensityWeighting=True,
    **kwargs,
):
    """
    Calculate the eSRRF3D temporal correlations for the given 3D image.

        img (ndarray): The input 4D image.
        mode (str, optional): The mode of time projection, default is "average", other option is "std".
        magnification_xy (float, optional): The magnification factor for the x and y axes. Default is 2.
        magnification_z (float, optional): The magnification factor for the z axis. Default is 2.
        radius (float, optional): The radius for the xy plane. Default is 1.5.
        sensitivity (float, optional): The sensitivity for the calculation. Default is 1.
        voxel_ratio (float, optional): The ratio of voxel dimensions (z to xy). Default is 4.
        doIntensityWeighting (bool, optional): Whether to perform intensity weighting. Default is True.
        **kwargs: Additional parameters for the eSRRF3D calculation, including:
            - radius_z (float, optional): The radius for the z axis.
            - run_type (str, optional): The type of the run.
            - keep_gradients (bool, optional): Whether to keep the gradients.
            - keep_interpolated (bool, optional): Whether to keep the interpolated values.
            - correlation (str, optional): The type of correlation to use.
            - framewindow (int, optional): The window size for frames.
            - rollingoverlap (int, optional): The overlap size for rolling.

        ndarray: The calculated eSRRF3D temporal correlations.
    """
    esrrf_calculator = eSRRF3D()
    return esrrf_calculator.run(img, **kwargs)
