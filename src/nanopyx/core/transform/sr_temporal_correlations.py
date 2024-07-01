import numpy as np


def calculate_SRRF_temporal_correlations(im: np.array, order: int = 1, do_integrate_lag_times: bool = 0):
    """
    Calculate temporal correlations for Super-Resolution Radial Fluctuations (SRRF).

    Args:
        im (np.array): Input 3D numpy array containing temporal data.
        order (int, optional): Order of temporal correlation. Should be 0, 1, -1, 2, 3, or 4.
            Defaults to 1.
        do_integrate_lag_times (bool, optional): Whether to integrate lag times.
            Defaults to False (0).

    Returns:
        np.array: Calculated temporal correlations based on the specified order and integration.

    Raises:
        AssertionError: If the input array doesn't have 3 dimensions or if the order is greater than 4.

    Note:
        - If `order` is 0, the maximum value over time is calculated.
        - If `order` is 1, the mean value over time is calculated.
        - If `order` is -1, the pairwise product sum is calculated.
        - If `order` is 2, 3, or 4, more advanced calculations are performed based on `do_integrate_lag_times`.

    """
    im = np.array(im, dtype="float32")
    assert im.ndim == 3 and order <= 4

    if order == 0:  # TODO: check order number in NanoJ
        out_array = np.amax(im, axis=0)

    elif order == 1:
        out_array = np.mean(im, axis=0)

    elif order == -1:
        out_array = calculate_pairwise_product_sum(im)

    else:  # order = 2 or order = 3 or order = 4
        out_array = calculate_acrf_(im, order, do_integrate_lag_times)

    return out_array


def calculate_eSRRF_temporal_correlations(im: np.array, correlation: str):
    """
    Calculate temporal correlations for enhanced Super-Resolution Radial Fluctuations (eSRRF).

    Args:
        im (np.array): Input 3D numpy array containing temporal data.
        correlation (str): Type of correlation to calculate. Should be "AVG", "VAR", or "TAC2".

    Returns:
        np.array: Calculated temporal correlations based on the specified correlation type.

    Raises:
        ValueError: If the specified correlation type is not "AVG", "VAR", or "TAC2".

    """
    im = np.array(im, dtype="float32")

    if correlation == "AVG":
        out_array = np.mean(im, axis=0)

    elif correlation == "VAR":
        out_array = np.var(im, axis=0)

    elif correlation == "TAC2":
        out_array = calculate_tac2(im)

    else:
        raise ValueError(f"Type of correlation must be AVG, VAR or TAC2")

    return out_array


def calculate_pairwise_product_sum(rad_array):
    """
    Calculate pairwise product sum of a 3D numpy array.

    Args:
        rad_array (np.array): Input 3D numpy array.

    Returns:
        np.array: Pairwise product sum of the input array.

    """
    n_time_points, height_m, width_m = rad_array.shape

    out_array = np.zeros((height_m, width_m), dtype=np.float32)
    # max_array = np.max(rad_array, axis=0)
    counter = 0
    pps = 0

    for t0 in range(n_time_points):
        r0 = np.maximum(rad_array[t0], 0)
        if np.any(r0) > 0:
            for t1 in range(t0, n_time_points):
                r1 = np.maximum(rad_array[t1], 0)
                pps += r0 * r1
                counter += 1
        else:
            counter += n_time_points - t0
    pps = pps / max(counter, 1)
    out_array = pps

    return out_array


def calculate_acrf_(rad_array, order, do_integrate_lag_times):
    """
    Calculate Auto-Correlation Radial Fluctuations (ACRF) for temporal data.

    Args:
        rad_array (np.array): Input 3D numpy array containing temporal data.
        order (int): Order of ACRF calculation.
        do_integrate_lag_times (bool): Whether to integrate lag times.

    Returns:
        np.array: Calculated ACRF based on the specified order and integration.

    """
    im = rad_array.copy()
    n_time_points, height_m, width_m = im.shape
    mean = np.mean(im, axis=0)

    out_array = np.zeros((height_m, width_m), dtype=np.float32)

    abcd = np.zeros((height_m, width_m), dtype=np.float32)
    abc = np.zeros((height_m, width_m), dtype=np.float32)
    ab = np.zeros((height_m, width_m), dtype=np.float32)
    cd = np.zeros((height_m, width_m), dtype=np.float32)
    ac = np.zeros((height_m, width_m), dtype=np.float32)
    bd = np.zeros((height_m, width_m), dtype=np.float32)
    ad = np.zeros((height_m, width_m), dtype=np.float32)
    bc = np.zeros((height_m, width_m), dtype=np.float32)

    if do_integrate_lag_times != 1:
        t = 0
        while t < n_time_points - order:
            ab = ab + (im[t] - mean) * (im[t + 1] - mean)
            if order == 3:
                abc = abc + (im[t] - mean) * (im[t + 1] - mean) + (im[t + 2] - mean)
            if order == 4:
                a = im[t] - mean
                b = im[t + 1] - mean
                c = im[t + 2] - mean
                d = im[t + 3] - mean
                abcd = abcd + np.multiply(np.multiply(np.multiply(a, b), c), d)
                cd = cd + np.multiply(c, d)
                ac = ac + np.multiply(a, c)
                bd = bd + np.multiply(b, d)
                ad = ad + np.multiply(a, d)
                bc = bc + np.multiply(b, c)
            t = t + 1
        if order == 3:
            out_array = np.absolute(abc) / n_time_points
        elif order == 4:
            out_array = np.absolute(abcd - ab * cd - ac * bd - ad * bc) / n_time_points
        else:
            out_array = np.absolute(ab) / n_time_points

    else:
        n_binned_time_points = n_time_points
        tbin = 0
        while n_binned_time_points > order:
            t = 0
            ab = np.zeros((height_m, width_m), dtype=np.float32)
            while t < n_binned_time_points - order:
                tbin = t * order
                ab = ab + (im[t] - mean) * (im[t + 1] - mean)
                if order == 3:
                    abc = abc + (im[t] - mean) * (im[t + 1] - mean) + (im[t + 2] - mean)
                if order == 4:
                    a = im[t] - mean
                    b = im[t + 1] - mean
                    c = im[t + 2] - mean
                    d = im[t + 3] - mean
                    abcd = abcd + np.multiply(np.multiply(np.multiply(a, b), c), d)
                    cd = cd + np.multiply(c, d)
                    ac = ac + np.multiply(a, c)
                    bd = bd + np.multiply(b, d)
                    ad = ad + np.multiply(a, d)
                    bc = bc + np.multiply(b, c)

                im[t] = np.zeros((height_m, width_m), dtype=np.float32)

                if tbin < n_binned_time_points:
                    for _t in range(order - 1):
                        im[t] = im[t] + np.divide(im[tbin + _t], order)
                t = t + 1
            if order == 3:
                out_array = np.absolute(abc) / n_binned_time_points
            elif order == 4:
                out_array = np.absolute(abcd - ab * cd - ac * bd - ad * bc) / n_binned_time_points
            else:
                out_array = np.absolute(ab) / n_binned_time_points

            n_binned_time_points = n_binned_time_points / order

    return out_array


def calculate_tac2(rad_array):
    """
    Calculate Temporal Autocorrelation 2 (TAC2) for temporal data.

    Args:
        rad_array (np.array): Input 3D numpy array containing temporal data.

    Returns:
        np.array: Calculated TAC2.

    """
    mean = np.mean(rad_array, axis=0)
    centered = rad_array - mean  # center data around the mean
    nlag = 1  # number of lags to compute TAC2 for
    out_array = np.mean(centered[:-nlag] * centered[nlag:], axis=0)

    return out_array


def calculate_eSRRF3d_temporal_correlations(
    rgc_map, correlation: str = "AVG", framewindow: float = 5, rollingoverlap: float = 2
):
    # correlation (str): Type of correlation to calculate. Should be "AVG", "VAR", or "TAC2"
    n_frames, n_slices, n_rows, n_cols = rgc_map.shape[0], rgc_map.shape[1], rgc_map.shape[2], rgc_map.shape[3]

    if n_frames == 1:
        print("Only one frame, no temporal correlations can be calculated")
        return rgc_map

    else:
        if framewindow > 0:
            if rollingoverlap:
                n_windows = int(n_frames / (framewindow - rollingoverlap))
            else:
                n_windows = int(n_frames / framewindow)
                rollingoverlap = 0
        else:
            n_windows = 1
            framewindow = n_frames - 1

        avg_rgc_map = np.zeros((n_windows, n_slices, n_rows, n_cols), dtype=np.float32)

        for w in range(n_windows):
            start_frame = w * (int(framewindow) - int(rollingoverlap))
            end_frame = start_frame + int(framewindow)
            avg_rgc_map[w, :, :, :] = calculate_eSRRF_temporal_correlations(
                rgc_map[start_frame:end_frame, :, :, :], correlation
            )

        return avg_rgc_map
