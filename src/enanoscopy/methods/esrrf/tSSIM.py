from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import numpy as np
import cv2
# import argparse

# skimage.metrics.structural_similarity(im1, im2, *, win_size=None, gradient=False, data_range=None, channel_axis=None, multichannel=False, gaussian_weights=False, full=False, **kwargs)
# Parameters:
# * im1, im2 - ndarray: Images. Any dimensionality with same shape.
#
# * win_size - int or None, optional: The side-length of the sliding window used in comparison. Must be an odd value.
# If gaussian_weights is True, this is ignored and the window size will depend on sigma.
#
# * gradient - bool, optional: If True, also return the gradient with respect to im2.
#
# * data_range - float, optional: The data range of the input image (distance between minimum and maximum possible values).
# By default, this is estimated from the image data-type.
#
# * channel_axis - int or None, optional: If None, the image is assumed to be a grayscale (single channel) image.
# Otherwise, this parameter indicates which axis of the array corresponds to channels.
#
# * multichannelbool, optional
# If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each
# channel then averaged. This argument is deprecated: specify channel_axis instead.
#
# * gaussian_weights - bool, optional: If True, each patch has its mean and variance spatially weighted by a normalized
# Gaussian kernel of width sigma=1.5.
#
# * full - bool, optional: If True, also return the full structural similarity image.

# Return:
# mssim - float: The mean structural similarity index over the image.
#
# grad - ndarray: The gradient of the structural similarity between im1 and im2 [2]. This is only returned if gradient is set to True.
#
# S - ndarray: The full SSIM image. This is only returned if full is set to True.
# Other Parameters:
# use_sample_covariance - bool: If True, normalize covariances by N-1 rather than, N where N is the number of pixels within the sliding window.
#
# K1 - float: Algorithm parameter, K1 (small constant, see [1]).
#
# K2 - float: Algorithm parameter, K2 (small constant, see [1]).
#
# sigma - float: Standard deviation for the Gaussian when gaussian_weights is True.

# define regularization method - data_range
def imgbitdepth(img):
    img_bitdepth = img.getbitd
    img_bitdepth = img.dtype(img)
     p.iinfo


def options():
    parser = argparse.ArgumentParser(description="Read image metadata")
    parser.add_argument("-o", "--first", help="Input image file.", required=True)
    parser.add_argument("-c", "--second", help="Input image file.", required=True)
    args = parser.parse_args()
    return args


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    mse_error /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE. The lower the error, the more "similar" the two images are.
    return mse_error


def compare(imageA, imageB):
    # Calculate the MSE and SSIM
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)

    # Return the SSIM. The higher the value, the more "similar" the two images are.
    return s


def main():
    # Get options
    args = options()

    # Import images
    image1 = cv2.imread(args.first)
    image2 = cv2.imread(args.second, 1)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Check for same size and ratio and report accordingly
    ho, wo, _ = image1.shape
    hc, wc, _ = image2.shape
    ratio_orig = ho / wo
    ratio_comp = hc / wc
    dim = (wc, hc)

    if round(ratio_orig, 2) != round(ratio_comp, 2):
        print("\nImages not of the same dimension. Check input.")
        exit()

    # Resize first image if the second image is smaller
    elif ho > hc and wo > wc:
        print("\nResizing original image for analysis...")
        gray1 = cv2.resize(gray1, dim)

    elif ho < hc and wo < wc:
        print("\nCompressed image has a larger dimension than the original. Check input.")
        exit()

    if round(ratio_orig, 2) == round(ratio_comp, 2):
        mse_value = mse(gray1, gray2)
        ssim_value = compare(gray1, gray2)
        print("MSE:", mse_value)
        print("SSIM:", ssim_value)


if __name__ == '__main__':
    main()