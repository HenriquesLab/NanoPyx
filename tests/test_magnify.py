import numpy as np
from nanopyx.core.transform.image_magnify import fourier_zoom

def test_fourier_zoom(random_image_with_ramp_squares):
    """
    """
    # image with random float values between -1000 and +1000
    image = random_image_with_ramp_squares
    z = 2

    zoomed_image = fourier_zoom(image, z)
    assert zoomed_image.shape[0] == image.shape[0] * z
    assert zoomed_image.shape[1] == image.shape[1] * z

    # the pixel values of the zoomed image at positions (0, 0), (0, z), (0,
    # 2*z), ..., (z, 0), (z, z), ... should be equal to the original image
    # values
    # np.testing.assert_allclose(zoomed_image[::z, ::z], image)