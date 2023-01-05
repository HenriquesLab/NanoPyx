import numpy as np
from skimage.transform import resize
from scipy.signal import fftconvolve


def estimate_shifts_dft(imgs, upsample_factor=1):
    """Estimate shifts using DFT.
    Parameters
    ----------
    imgs : ndarray
        3D data cube containing the registered images.
    upsample_factor : int
        Upsampling factor. Images will be registered to within
        1 / upsample_factor of a pixel. Images must be registered
        to within 1 pixel before calling this function.
    Returns
    -------
    shifts_r : ndarray
        Estimated row shifts.
    shifts_c : ndarray
        Estimated column shifts.
    """

    # The code below does the following:
    # 1. Upsamples the image by a factor of M (i.e. the pixel size is reduced by a factor of 10)
    # 2. Computes the cross-correlation between each pair of images
    # 3. Computes the phase-correlation from the cross-correlation
    # 4. Finds the maximum of the phase-correlation
    # 5. Computes the shift by finding the difference between the maximum and the center of the image
    # 6. Downsamples the shift by a factor of M (i.e. the shift is converted back to the original pixel size)

    if upsample_factor != 1:
        imgs = np.stack([resize(img, (int(np.ceil(img.shape[0] * upsample_factor)),
                                      int(np.ceil(img.shape[1] * upsample_factor))),
                                order=1, preserve_range=True, mode='reflect')
                         for img in imgs], axis=0)
    # Compute cross-correlation
    num_frames, num_rows, num_cols = imgs.shape
    cross_correlation = np.zeros((num_rows, num_cols, num_frames, num_frames))
    for m in range(num_frames):
        for n in range(num_frames):
            cross_correlation[:, :, m, n] = fftconvolve(imgs[m], imgs[n][::-1, ::-1], mode='same')
    # Compute phase correlation
    cross_correlation /= np.abs(cross_correlation)
    cross_correlation = np.fft.fftshift(cross_correlation, axes=(0, 1))
    phase_correlation = np.fft.ifft2(cross_correlation).real
    # Locate maxima
    shifts = np.array(np.unravel_index(np.argmax(phase_correlation, axis=None),
                                       phase_correlation.shape)).T - np.array(phase_correlation.shape) // 2
    shifts = np.roll(shifts, 1, axis=1)
    if upsample_factor != 1:
        shifts = shifts / upsample_factor
    return shifts[:, 0], shifts[:, 1]


def minimize_shifts(shifts_r, shifts_c):
    """Find the optimal shifts to minimize the residual error between images.
    Args:
        shifts_r (ndarray): Estimated row shifts.
        shifts_c (ndarray): Estimated column shifts.
    Returns:
        shift_r (ndarray): The optimal shifts in the row direction.
        shift_c (ndarray): The optimal shifts in the column direction.
    """
    
    # The code below does the following:
    # 1. Finds the optimal shift in the row direction by minimizing the residual error between images
    # 2. Finds the optimal shift in the column direction by minimizing the residual error between images

    # Find the optimal shift in the row direction
    shift_r = np.zeros(shifts_r.shape[0])
    for m in range(shifts_r.shape[0]):
        shift_r[m] = shifts_r[m, m]
        for n in range(shifts_r.shape[1]):
            if n != m:
                shift_r[m] += shifts_r[m, n] - shifts_r[n, m]
    shift_r /= shifts_r.shape[0]
    # Find the optimal shift in the column direction
    shift_c = np.zeros(shifts_c.shape[0])
    for m in range(shifts_c.shape[0]):
        shift_c[m] = shifts_c[m, m]
        for n in range(shifts_c.shape[1]):
            if n != m:
                shift_c[m] += shifts_c[m, n] - shifts_c[n, m]
    shift_c /= shifts_c.shape[0]
    return shift_r, shift_c
