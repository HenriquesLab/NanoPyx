import numpy as np

from nanopyx.core.transform.image_polar import (
    bicubic_polar,
    catmull_rom_polar,
    lanczos_polar,
)

from nanopyx.core.transform.image_cart import (
    bicubic_cart,
    catmull_rom_cart,
    lanczos_cart,
)

def test_bicubic_polar(random_image_with_squares, plt):
    image = random_image_with_squares

    polar = bicubic_polar(image)
    cartesian = bicubic_cart(polar, image.shape[1], image.shape[0])
    delta = image - cartesian

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(polar)
    axarr[2].imshow(cartesian)
    axarr[3].imshow(delta)


def test_catmull_rom_polar(random_image_with_squares, plt):
    image = random_image_with_squares

    polar = catmull_rom_polar(image)
    cartesian = catmull_rom_cart(polar, image.shape[1], image.shape[0])
    delta = image - cartesian

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(polar)
    axarr[2].imshow(cartesian)
    axarr[3].imshow(delta)


def test_lanczos_polar(random_image_with_squares, plt):
    image = random_image_with_squares

    polar = lanczos_polar(image)
    cartesian = lanczos_cart(polar, image.shape[1], image.shape[0])
    delta = image - cartesian

    plt.figure()
    f, axarr = plt.subplots(1, 4)
    axarr[0].imshow(image)
    axarr[1].imshow(polar)
    axarr[2].imshow(cartesian)
    axarr[3].imshow(delta)
