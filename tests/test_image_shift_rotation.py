import numpy as np
from nanopyx.core.transform.image_shift_rotate import (  # scipy_rotate_shift,
    bicubic_rotate_shift,
    catmull_rom_rotate_shift,
    cv2_rotate_shift,
    lanczos_rotate_shift,
    skimage_rotate_shift,
)


def test_bicubic_rotate_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi / 10

    w = image.shape[1]
    h = image.shape[0]
    cx = np.random.randint(0, w) / 10
    cy = np.random.randint(0, h) / 10

    shifted1 = bicubic_rotate_shift(image, theta, cx, cy)
    shifted2 = bicubic_rotate_shift(shifted1, -theta, cx, cy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_catmull_rom_rotate_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi / 10

    w = image.shape[1]
    h = image.shape[0]
    cx = np.random.randint(0, w) / 10
    cy = np.random.randint(0, h) / 10

    shifted1 = catmull_rom_rotate_shift(image, theta, cx, cy)
    shifted2 = catmull_rom_rotate_shift(shifted1, -theta, cx, cy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_cv2_rotate_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi / 10

    w = image.shape[1]
    h = image.shape[0]
    cx = np.random.randint(0, w) / 10
    cy = np.random.randint(0, h) / 10

    shifted1 = cv2_rotate_shift(image, theta, cx, cy)
    shifted2 = cv2_rotate_shift(shifted1, -theta, cx, cy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_lanczos_rotate_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi / 10

    w = image.shape[1]
    h = image.shape[0]
    cx = np.random.randint(0, w) / 10
    cy = np.random.randint(0, h) / 10

    shifted1 = lanczos_rotate_shift(image, theta, cx, cy)
    shifted2 = lanczos_rotate_shift(shifted1, -theta, cx, cy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


# def test_scipy_rotate_shift(random_image_with_squares, plt):


def test_skimage_rotate_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    theta = np.random.rand() * np.pi / 10

    w = image.shape[1]
    h = image.shape[0]
    cx = np.random.randint(0, w) / 10
    cy = np.random.randint(0, h) / 10

    shifted1 = skimage_rotate_shift(image, theta, cx, cy)
    shifted2 = skimage_rotate_shift(shifted1, -theta, cx, cy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)
