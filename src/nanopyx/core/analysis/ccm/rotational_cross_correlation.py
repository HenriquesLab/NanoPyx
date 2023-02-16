import numpy as np

from .ccm import calculate_rccm, calculate_ccm_polar 
from .estimate_shift import GetMaxOptimizer

def calculate_rotation_shift(img_slice, img_ref, method="subpixel"):

    rccm = calculate_rccm(img_slice, img_ref)

    width = rccm.shape[2]
    height = rccm.shape[1]

    y_shift = None
    x_shift = None
    ang_shift = None
    max_sim = -np.inf

    for deg in range(360):
        slice_ccm = rccm[deg,:,:]

        max_coords, ccmval = calculate_peak(slice_ccm, method)

        if ccmval > max_sim:
            y_shift = (height/2.0 - max_coords[0] - 1)
            x_shift = (width/2.0 - max_coords[1] - 1)
            ang_shift = np.deg2rad(deg)
            max_sim = ccmval
    
    return y_shift, x_shift, ang_shift, max_sim

def calculate_rotation_polar(img_slice, img_ref, method="subpixel"):

    rccm_polar = calculate_ccm_polar(img_slice, img_ref)

    max_coords, maxsim = calculate_peak(rccm_polar, method)

    theta = max_coords[0] * np.pi/180
    r = max_coords[1]

    return theta, r, maxsim

def calculate_peak(ccm, method="subpixel"):

    if method == "subpixel":
        optimizer = GetMaxOptimizer(ccm)
        max_coords = optimizer.get_max()
        ccm_max_value = -optimizer.get_interpolated_px_value(max_coords)
    else:
        max_coords = np.unravel_index(ccm.argmax(), ccm.shape)
        ccm_max_value = slice_ccm[max_coords[0], max_coords[1]]

    return max_coords, ccm_max_value