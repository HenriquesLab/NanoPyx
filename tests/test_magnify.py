import numpy as np

from nanopyx.core.transform.binning import rebin_2d
from nanopyx.core.transform.image_magnify import (bicubic_zoom,
                                                  catmull_rom_zoom, cv2_zoom,
                                                  fourier_zoom, lanczos_zoom,
                                                  scipy_zoom, skimage_zoom)


def test_fourier_zoom(random_image_with_ramp, plt):
    imageDownsampled = rebin_2d(random_image_with_ramp, 2, mode="mean")
    assert imageDownsampled.shape[0] == random_image_with_ramp.shape[0] / 2
    assert imageDownsampled.shape[1] == random_image_with_ramp.shape[1] / 2

    imageMagnified = fourier_zoom(imageDownsampled, 2)
    assert imageMagnified.shape[0] == random_image_with_ramp.shape[0]
    assert imageMagnified.shape[1] == random_image_with_ramp.shape[1]
    np.testing.assert_allclose(imageMagnified, random_image_with_ramp, rtol=10, atol=10)

    plt.figure()
    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(random_image_with_ramp)
    axarr[1].imshow(imageMagnified)
    axarr[2].imshow(random_image_with_ramp-imageMagnified)


def test_catmull_rom_zoom(random_image_with_ramp, plt):
    imageDownsampled = rebin_2d(random_image_with_ramp, 2, mode="mean")
    assert imageDownsampled.shape[0] == random_image_with_ramp.shape[0] / 2
    assert imageDownsampled.shape[1] == random_image_with_ramp.shape[1] / 2

    imageMagnified = catmull_rom_zoom(imageDownsampled, 2)
    assert imageMagnified.shape[0] == random_image_with_ramp.shape[0]
    assert imageMagnified.shape[1] == random_image_with_ramp.shape[1]
    np.testing.assert_allclose(imageMagnified, random_image_with_ramp, rtol=10, atol=10)

    plt.figure()
    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(random_image_with_ramp)
    axarr[1].imshow(imageMagnified)
    axarr[2].imshow(random_image_with_ramp-imageMagnified)


def test_lanczos_zoom(random_image_with_ramp, plt):
    imageDownsampled = rebin_2d(random_image_with_ramp, 2, mode="mean")
    assert imageDownsampled.shape[0] == random_image_with_ramp.shape[0] / 2
    assert imageDownsampled.shape[1] == random_image_with_ramp.shape[1] / 2

    imageMagnified = lanczos_zoom(imageDownsampled, 2)
    assert imageMagnified.shape[0] == random_image_with_ramp.shape[0]
    assert imageMagnified.shape[1] == random_image_with_ramp.shape[1]
    np.testing.assert_allclose(imageMagnified, random_image_with_ramp, rtol=10, atol=10)

    plt.figure()
    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(random_image_with_ramp)
    axarr[1].imshow(imageMagnified)
    axarr[2].imshow(random_image_with_ramp-imageMagnified)


def test_bicubic_zoom(random_image_with_ramp, plt):
    imageDownsampled = rebin_2d(random_image_with_ramp, 2, mode="mean")
    assert imageDownsampled.shape[0] == random_image_with_ramp.shape[0] / 2
    assert imageDownsampled.shape[1] == random_image_with_ramp.shape[1] / 2

    imageMagnified = bicubic_zoom(imageDownsampled, 2)
    assert imageMagnified.shape[0] == random_image_with_ramp.shape[0]
    assert imageMagnified.shape[1] == random_image_with_ramp.shape[1]
    np.testing.assert_allclose(imageMagnified, random_image_with_ramp, rtol=10, atol=10)

    plt.figure()
    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(random_image_with_ramp)
    axarr[1].imshow(imageMagnified)
    axarr[2].imshow(random_image_with_ramp-imageMagnified)


def test_scipy_zoom(random_image_with_ramp, plt):
    imageDownsampled = rebin_2d(random_image_with_ramp, 2, mode="mean")
    assert imageDownsampled.shape[0] == random_image_with_ramp.shape[0] / 2
    assert imageDownsampled.shape[1] == random_image_with_ramp.shape[1] / 2

    imageMagnified = scipy_zoom(imageDownsampled, 2)
    assert imageMagnified.shape[0] == random_image_with_ramp.shape[0]
    assert imageMagnified.shape[1] == random_image_with_ramp.shape[1]
    np.testing.assert_allclose(imageMagnified, random_image_with_ramp, rtol=10, atol=10)

    plt.figure()
    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(random_image_with_ramp)
    axarr[1].imshow(imageMagnified)
    axarr[2].imshow(random_image_with_ramp-imageMagnified)

def test_skimage_zoom(random_image_with_ramp, plt):
    imageDownsampled = rebin_2d(random_image_with_ramp, 2, mode="mean")
    assert imageDownsampled.shape[0] == random_image_with_ramp.shape[0] / 2
    assert imageDownsampled.shape[1] == random_image_with_ramp.shape[1] / 2

    imageMagnified = skimage_zoom(imageDownsampled, 2)
    assert imageMagnified.shape[0] == random_image_with_ramp.shape[0]
    assert imageMagnified.shape[1] == random_image_with_ramp.shape[1]
    np.testing.assert_allclose(imageMagnified, random_image_with_ramp, rtol=10, atol=10)

    plt.figure()
    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(random_image_with_ramp)
    axarr[1].imshow(imageMagnified)
    axarr[2].imshow(random_image_with_ramp-imageMagnified)


def test_cv2_zoom(random_image_with_ramp, plt):
    imageDownsampled = rebin_2d(random_image_with_ramp, 2, mode="mean")
    assert imageDownsampled.shape[0] == random_image_with_ramp.shape[0] / 2
    assert imageDownsampled.shape[1] == random_image_with_ramp.shape[1] / 2

    imageMagnified = cv2_zoom(imageDownsampled, 2)
    assert imageMagnified.shape[0] == random_image_with_ramp.shape[0]
    assert imageMagnified.shape[1] == random_image_with_ramp.shape[1]
    np.testing.assert_allclose(imageMagnified, random_image_with_ramp, rtol=10, atol=10)

    plt.figure()
    f, axarr = plt.subplots(1,3) 
    axarr[0].imshow(random_image_with_ramp)
    axarr[1].imshow(imageMagnified)
    axarr[2].imshow(random_image_with_ramp-imageMagnified)