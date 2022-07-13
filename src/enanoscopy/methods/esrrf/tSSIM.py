from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import skimage.io as io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#  import cv2
#  import argparse

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
#print("start")

# open image
# define regularization method - data_range
# def imgbitdepth(img):
#     img_bitdepth = img.getbitd
#     img_bitdepth = img.dtype(img)
#      p.iinfo

class tSSIM(object):
    def __init__(self):
        self.reg_method = 1  # Reg-Method: 0-data range, 1-bit depth
        self.block_size = 500  # Block size: 0-auto, >0 custom.  -> Number of blocks for rolling window analysis
        self.sigma_cut = 3  # Sigma cut off: default = 3
        self.cutoff_over_time = False  # Calculate cutoff over time, default = False
        self.smooth_factor = 0.15  # Smoothing factor, default = 0.15

    def regularization(imageB, reg_method):
        if int(reg_method) == 1: #reg_method is image bit depth
           reg_range=2**(imageB.dtype.itemsize*8)
        else: # reg_method is image dynamic range between 1% and 99% pixels value
               imageB_values=np.sort(imageB, axis=None)
               reg_range = imageB_values[round(imageB_values.size * 0.99), ] - imageB_values[round(imageB_values.size * 0.01), ]
        return reg_range

    def tempssim(img, reg_method):
        reg_value = tSSIM.regularization(img, reg_method)
        # Calculate the SSIM
        s = np.zeros((1, img.shape[0]-1))
        for t in range(img.shape[0]-1):
            s[0, t] = ssim(img[t], img[t+1], data_range=reg_value)
        # Return the SSIM. The higher the value, the more "similar" the two images are.
        return s

    def rollingtempssim(img, block_size, reg_method):

        n_frames = img.shape[0]  # Number of frames in the stack

        if block_size <= 0 or block_size >= (n_frames-1):
            n_block = 1

        elif block_size > 0 and block_size < (n_frames-1):
            n_block = n_frames - block_size -1

        ssim_block = np.zeros((n_block, block_size))  ### !!! Block size !!!
        for i in range(n_block):
            ssim_block[i,:]=tSSIM.tempssim(img[i*block_size:(i*block_size)+n_frames-1, :, :], reg_method)

        return ssim_block

    #def main():

img = io.imread("/Users/hannahheil/Desktop/testimages/Set03_COS7_PrSS-mEmerald-KDEL.tif", as_gray=False)
print(img.shape)
print(img.shape[0])

out = tSSIM.rollingtempssim(img, 500, 1)
frame = np.arange(1,500)
print(frame)
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(frame, out[1]);  # Plot some data on the axes.




    #if __name__ == '__main__':
    #    main()