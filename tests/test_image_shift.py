import numpy as np

from nanopyx.core.transform.image_shift import (
    bicubic_shift,
    catmull_rom_shift,
    cv2_shift,
    lanczos_shift,
    scipy_shift,
    skimage_shift,
)


def test_catmull_rom_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    w = image.shape[1]
    h = image.shape[0]
    dx = np.random.randint(0, w) / 4
    dy = np.random.randint(0, h) / 4

    shifted1 = catmull_rom_shift(image, dx, dy)
    shifted2 = catmull_rom_shift(shifted1, -dx, -dy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_bicubic_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    w = image.shape[1]
    h = image.shape[0]
    dx = np.random.randint(0, w) / 4
    dy = np.random.randint(0, h) / 4

    shifted1 = bicubic_shift(image, dx, dy)
    shifted2 = bicubic_shift(shifted1, -dx, -dy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_lanczos_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    w = image.shape[1]
    h = image.shape[0]
    dx = np.random.randint(0, w) / 4
    dy = np.random.randint(0, h) / 4

    shifted1 = lanczos_shift(image, dx, dy)
    shifted2 = lanczos_shift(shifted1, -dx, -dy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_scipy_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    w = image.shape[1]
    h = image.shape[0]
    dx = np.random.randint(0, w) / 4
    dy = np.random.randint(0, h) / 4

    shifted1 = scipy_shift(image, dx, dy)
    shifted2 = scipy_shift(shifted1, -dx, -dy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_skimage_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    w = image.shape[1]
    h = image.shape[0]
    dx = np.random.randint(0, w) / 4
    dy = np.random.randint(0, h) / 4

    shifted1 = skimage_shift(image, dx, dy)
    shifted2 = skimage_shift(shifted1, -dx, -dy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_cv2_shift(random_image_with_squares, plt):
    image = random_image_with_squares
    w = image.shape[1]
    h = image.shape[0]
    dx = np.random.randint(0, w) / 4
    dy = np.random.randint(0, h) / 4

    shifted1 = cv2_shift(image, dx, dy)
    shifted2 = cv2_shift(shifted1, -dx, -dy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)


def test_cv2_shift_2(random_image_with_squares, plt):
    image = random_image_with_squares
    w = image.shape[1]
    h = image.shape[0]
    dx = np.random.randint(0, w) / 4
    dy = np.random.randint(0, h) / 4

    shifted1 = cv2_shift(image, dx, dy)
    shifted2 = cv2_shift(shifted1, -dx, -dy)
    delta = image - shifted2
    assert shifted1.shape == shifted2.shape == image.shape

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(shifted1)
    axarr[2].imshow(shifted2)
    axarr[3].imshow(delta)
