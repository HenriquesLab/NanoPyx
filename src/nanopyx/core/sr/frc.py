# TODO: fix this

import math
from math import pi

import numpy as np
from scipy.interpolate import UnivariateSpline

# REF: based on https://c4science.ch/source/ijp-frc/browse/master/src/main/java/ch/epfl/biop/frc/FRC.java


perimeter_sampling_factor = 1
use_half_circle = True


class ThresholdMethod:
    FIXED_1_OVER_7 = "Fixed 1/7"
    HALF_BIT = "Half-bit"
    THREE_SIGMA = "Three sigma"


def pad(im, width, height):
    im2 = np.zeros((height, width), dtype=im.dtype)
    im2[: im.shape[0], : im.shape[1]] = im
    return im2


def calculate_frc_curve(im1, im2):
    # Pad images to the same size
    max_width = max(im1.shape[1], im2.shape[1])
    max_height = max(im1.shape[0], im2.shape[0])
    im1 = pad(im1, max_width, max_height)
    im2 = pad(im2, max_width, max_height)

    # Calculate the complex Fourier transform images
    fft1 = np.fft.fftn(im1)
    fft2 = np.fft.fftn(im2)

    # Calculate the power spectrum images
    ps1 = np.abs(fft1) ** 2
    ps2 = np.abs(fft2) ** 2

    # Calculate the cross-power spectrum
    cps = fft1 * np.conj(fft2)

    # Calculate the cross-correlation image
    cc = np.fft.ifftn(cps)

    # Calculate the 1D cross-correlation curve
    cr = np.abs(cc)

    # Calculate the radii of the samples
    max_radius = int(cr.shape[0] / 2)
    if use_half_circle:
        max_radius = int(max_radius / 2)
    radii = np.arange(0, max_radius)
    radii *= perimeter_sampling_factor * pi

    # Calculate the FRC curve
    frc = np.zeros((len(radii), 3))
    for i, radius in enumerate(radii):
        samples = _samples_at_radius(cr, radius)
        frc[i][0] = radius
        frc[i][1] = np.mean(samples)
        frc[i][2] = len(samples)

    return frc


def _samples_at_radius(image, radius):
    x, y = np.ogrid[: image.shape[0], : image.shape[1]]
    center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return image[
        np.where(
            (distance_from_center > radius - 0.5)
            & (distance_from_center < radius + 0.5)
        )
    ]


def get_smoothed_curve(frc_curve):
    s_curve = frc_curve.copy()
    x_vals = frc_curve[:][0]
    y_vals = frc_curve[:][1]

    # Fit a LOESS curve to the data using the UnivariateSpline class
    loess = UnivariateSpline(x_vals, y_vals, k=1, s=0.1)

    # Evaluate the LOESS curve at the interpolated x values
    y_smoothed = loess(x_vals)

    s_curve[:][1] = y_smoothed

    return s_curve


def calculate_threshold_curve(frc_curve: np.ndarray, method):
    threshold = np.zeros(frc_curve.shape[0])
    for i in range(threshold.shape[0]):
        if method == ThresholdMethod.HALF_BIT:
            threshold[i] = (0.2071 * math.sqrt(frc_curve[i][2]) + 1.9102) / (
                    1.2071 * math.sqrt(frc_curve[i][2]) + 0.9102
            )
        elif method == ThresholdMethod.THREE_SIGMA:
            threshold[i] = 3.0 / math.sqrt(frc_curve[i][2] / 2.0)
        else:
            threshold[i] = 0.1428
    return threshold


def get_intersections(frc_curve: np.ndarray, threshold_curve: np.ndarray):
    if len(frc_curve) != len(threshold_curve):
        print(
            "Error: Unable to calculate FRC curve intersections due to input length mismatch."
        )
        return None

    intersections = np.zeros(len(frc_curve) - 1)
    count = 0
    for i in range(1, len(frc_curve)):
        y1 = frc_curve[i - 1][1]
        y2 = frc_curve[i][1]
        y3 = threshold_curve[i - 1]
        y4 = threshold_curve[i]
        if not ((y3 >= y1 and y4 < y2) or (y1 >= y3 and y2 < y4)):
            continue
        x1 = frc_curve[i - 1][0]
        x2 = frc_curve[i][0]
        x3 = x1
        x4 = x2
        x1_x2 = x1 - x2
        x3_x4 = x3 - x4
        y1_y2 = y1 - y2
        y3_y4 = y3 - y4
        if x1_x2 * y3_y4 - y1_y2 * x3_x4 == 0:
            if y1 == y3:
                intersections[count] = x1
                count += 1
        else:
            px = ((x1 * y2 - y1 * x2) * x3_x4 - x1_x2 * (x3 * y4 - y3 * x4)) / (
                x1_x2 * y3_y4 - y1_y2 * x3_x4
            )
            if px >= x1 and px < x2:
                intersections[count] = px
                count += 1
    return np.copy(intersections[:count])


def get_correct_intersection(intersections: np.ndarray, method):
    if intersections is None or intersections.shape[0] == 0:
        return 0
    if method == ThresholdMethod.HALF_BIT or method == ThresholdMethod.THREE_SIGMA:
        if intersections.shape[0] > 1:
            return intersections[1]
        else:
            return intersections[0]
    return intersections[0]


# Cache the Tukey window function
taper_x = np.array([])
taper_y = np.array([])


def get_square_tapered_image(data_image):
    # Use a Tukey window function
    global taper_x, taper_y
    taper_x = get_window_function(taper_x, data_image.shape[0])
    taper_y = get_window_function(taper_y, data_image.shape[1])

    size = max(data_image.shape[0], data_image.shape[1])

    # Pad to a power of 2
    new_size = 0
    for i in range(4, 15):
        new_size = 2**i
        if size <= new_size:
            break

    if size > new_size:
        return None  # Error

    data_image = data_image.astype(np.float32)
    data = data_image.flatten()
    pixels = np.empty((new_size, new_size), dtype=np.float32)
    # Note that the limits at 0 and size-1 the taper is zero so this can be ignored
    maxx_1 = data_image.shape[0] - 1
    maxy_1 = data_image.shape[1] - 1
    old_width = data_image.shape[0]

    for y in range(1, maxy_1):
        y_tmp = taper_y[y]
        for x in range(1, maxx_1):
            i = y * old_width + x
            ii = y * new_size + x
            pixels[ii] = data[i] * taper_x[x] * y_tmp

    return pixels


def get_window_function(taper: np.ndarray, size):
    if taper.shape[0] != size:
        # Re-use cached values
        global taper_x, taper_y
        if taper_x.shape[0] == size:
            return taper_x
        if taper_y.shape[1] == size:
            return taper_y

        boundary = size // 8
        upper_boundary = size - boundary
        taper = np.empty(size, dtype=np.float32)
        for i in range(size):
            if i < boundary or i > size - upper_boundary:
                taper[i] = np.sin(12.566370614359172 * i / size) ** 2
            else:
                taper[i] = 1
    return taper


def get_interpolated_values(x, y, images, maxx):
    xbase = int(x)
    ybase = int(y)
    x_fraction = x - xbase
    y_fraction = y - ybase
    if x_fraction < 0.0:
        x_fraction = 0.0
    if y_fraction < 0.0:
        y_fraction = 0.0

    lower_left_index = ybase * maxx + xbase
    lower_right_index = lower_left_index + 1
    upper_left_index = lower_left_index + maxx
    upper_right_index = upper_left_index + 1

    no_images = 3  # images.shape[0]

    values = np.empty(no_images)
    for i in range(no_images):
        image = images[i]
        lower_left = image[lower_left_index]
        lower_right = image[lower_right_index]
        upper_right = image[upper_left_index]
        upper_left = image[upper_right_index]

        upper_average = upper_left + x_fraction * (upper_right - upper_left)
        lower_average = lower_left + x_fraction * (lower_right - lower_left)
        values[i] = lower_average + y_fraction * (upper_average - lower_average)

    return values


def calculate_fire_number_from_FRC_curve(frc_curve: np.ndarray, method):
    threshold_curve = calculate_threshold_curve(frc_curve, method)
    intersections = get_intersections(frc_curve, threshold_curve)
    fire = None
    if intersections is not None and intersections.shape[0] != 0:
        spatial_frequency = get_correct_intersection(intersections, method)
        fire = 2 * (len(frc_curve) + 1) / spatial_frequency
    return fire


def calculate_fire_number(im1, im2, method):
    frc_curve = calculate_frc_curve(im1, im2)
    return calculate_fire_number_from_FRC_curve(frc_curve, method)


def do_plot(frc_curve, smooth_frc, tm, fire, name):
    import matplotlib.pyplot as plt

    # Prepare arrays for Plot class
    # Original FRC curve
    y = frc_curve[:][1]
    # Smoothed FRC Curve
    sy = smooth_frc[:][1]
    # Since the Fourier calculation only uses half of the image (from centre to the edge)
    # we must double the curve length to get the original maximum image width. In addition
    # the computation was up to the edge-1 pixels so add back a pixel to the curve length.
    # If we divide the value of the x axes by the highest spatial frequency (representing 1 pixel^-1)
    # we can get a calibrated frequency axis.
    x = frc_curve[:][0] / (2 * (frc_curve.shape[0] + 1))

    # Get Curve of Threshold
    # Curve representing the Threshold method calculation that should intercept with the FRC Curve
    thr_curve = calculate_threshold_curve(smooth_frc, tm)

    # Plot the data
    plt.figure()
    plt.title("FRC Of " + name)
    plt.xlabel("Spatial Frequency")
    plt.ylabel("Correlation")
    plt.ylim(0, 1)
    plt.xlim(0, x[-1])

    # Add Original Data in black
    plt.plot(x, y, "k-", linewidth=1)
    # Add Smoothed Curve in clear red
    plt.plot(x, sy, "r--", linewidth=1)
    # Add Threshold Curve in clear blue
    plt.plot(x, thr_curve, "b--", linewidth=1)
    # Add FIRE number line in gray
    plt.axvline(x=1 / fire, color="k", linestyle=":")
    # Add the FIRE number on the graph
    plt.text(0.02, 0.3, f"FIRE = {fire:.3f}")
    # Add legend to the plot
    plt.legend(["FRC", "Smoothed FRC", "Threshold"])

    return plt
