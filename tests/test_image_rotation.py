import numpy as np

from nanopyx.core.transform.image_rotate import (
    bicubic_rotate,
    catmull_rom_rotate,
    cv2_rotate,
    lanczos_rotate,
    scipy_rotate,
    skimage_rotate,
)

def test_bicubic_rotate(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi/10

    shifted1 = bicubic_rotate(image, theta)
    shifted2 = bicubic_rotate(shifted1, -theta)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_catmull_rom_rotate(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi/10

    shifted1 = catmull_rom_rotate(image, theta)
    shifted2 = catmull_rom_rotate(shifted1, -theta)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_cv2_rotate(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi/10

    shifted1 = cv2_rotate(image, theta)
    shifted2 = cv2_rotate(shifted1, -theta)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_lanczos_rotate(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi/10 

    shifted1 = lanczos_rotate(image, theta)
    shifted2 = lanczos_rotate(shifted1, -theta)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_scipy_rotate(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi/10

    shifted1 = scipy_rotate(image, theta)
    shifted2 = scipy_rotate(shifted1, -theta)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_skimage_rotate(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi/10

    shifted1 = skimage_rotate(image, theta)
    shifted2 = skimage_rotate(shifted1, -theta)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)
