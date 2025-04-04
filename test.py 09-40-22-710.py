import numpy as np
import tifffile

from nanopyx.core.transform import (
    RadialGradientConvergence,
    GradientRobertsCross,
    eSRRF_ST,
    CRShiftAndMagnify,
    Convolution2D,
)

from matplotlib import pyplot as plt


def sobel(img):

    kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    kernelx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)

    gx = Convolution2D().run(img, kernelx, run_type="OpenCL")
    gy = Convolution2D().run(img, kernely, run_type="OpenCL")

    return gx, gy


def testingkernels(img):

    kernelx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]).astype(np.float32)
    kernely = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]).astype(np.float32)

    print(kernelx, kernely)

    gx = Convolution2D().run(img, kernelx, run_type="OpenCL")
    gy = Convolution2D().run(img, kernely, run_type="OpenCL")

    return gx, gy


def robertscross(img):

    kernely = np.array([[0, -1], [1, 0]]).astype(np.float32)
    kernelx = np.array([[-1, 0], [0, 1]]).astype(np.float32)

    print(kernelx, kernely)

    gx = Convolution2D().run(img, kernelx, run_type="OpenCL")
    gy = Convolution2D().run(img, kernely, run_type="OpenCL")

    return gx, gy


if __name__ == "__main__":

    # Load image
    img = tifffile.imread(
        "/Users/bsaraiva/Code/NanoPyx/Frames_camera_resampled_10FPS.tif",
        key=range(1000),
    ).astype(np.float32)

    mag = 4
    grad_mag = 1
    radius = 1.5

    # # Calculate gradients
    # gx, gy = GradientRobertsCross().run(img, run_type="OpenCL")

    # Magnify image 5x
    imgmag = CRShiftAndMagnify().run(img, 0, 0, mag, mag, run_type="OpenCL")

    # # Magnify gradients 10x
    # gxmag = CRShiftAndMagnify().run(
    #     gx, 0, 0, mag * grad_mag, mag * grad_mag, run_type="OpenCL"
    # )
    # gymag = CRShiftAndMagnify().run(
    #     gy, 0, 0, mag * grad_mag, mag * grad_mag, run_type="OpenCL"
    # )

    # # Calculate radial gradient convergence
    # rgc_old = RadialGradientConvergence().run(
    #     gxmag,
    #     gymag,
    #     imgmag,
    #     doIntensityWeighting=True,
    #     magnification=mag,
    #     grad_magnification=grad_mag,
    #     radius=radius,
    #     offset=0.0,
    #     xyoffset=0,
    #     angle=0,
    #     sensitivity=1,
    #     run_type="opencl",
    # )

    # Now do the exact same but calculate the gradients using finite differences
    gx, gy = testingkernels(img)
    gxmag = CRShiftAndMagnify().run(
        gx, 0, 0, mag * grad_mag, mag * grad_mag, run_type="OpenCL"
    )
    gymag = CRShiftAndMagnify().run(
        gy, 0, 0, mag * grad_mag, mag * grad_mag, run_type="OpenCL"
    )
    rgc_fd_iw = RadialGradientConvergence().run(
        gxmag,
        gymag,
        imgmag,
        doIntensityWeighting=True,
        magnification=mag,
        grad_magnification=grad_mag,
        radius=radius,
        offset=0,
        xyoffset=0,
        angle=0,
        sensitivity=1,
        run_type="opencl",
    )

    rgc_fd = RadialGradientConvergence().run(
        gxmag,
        gymag,
        imgmag,
        doIntensityWeighting=False,
        magnification=mag,
        grad_magnification=grad_mag,
        radius=radius,
        offset=0,
        xyoffset=0,
        angle=0,
        sensitivity=1,
        run_type="opencl",
    )

    # Now do the exact same but calculate the gradients using finite differences
    # gx, gy = testingkernels(img)
    # gxmag = CRShiftAndMagnify().run(
    #     gx, 0, 0, mag * grad_mag, mag * grad_mag, run_type="OpenCL"
    # )
    # gymag = CRShiftAndMagnify().run(
    #     gy, 0, 0, mag * grad_mag, mag * grad_mag, run_type="OpenCL"
    # )
    # rgc_fd_05 = RadialGradientConvergence().run(
    #     gxmag,
    #     gymag,
    #     imgmag,
    #     doIntensityWeighting=True,
    #     offset=0.5,
    #     xyoffset=0,
    #     angle=0,
    #     sensitivity=1,
    #     run_type="opencl",
    # )

    # Now do the exact same but calculate the gradients using a robertscross
    gx, gy = robertscross(img)
    gxmag = CRShiftAndMagnify().run(
        gx, 0, 0, mag * grad_mag, mag * grad_mag, run_type="OpenCL"
    )
    gymag = CRShiftAndMagnify().run(
        gy, 0, 0, mag * grad_mag, mag * grad_mag, run_type="OpenCL"
    )

    offset = 0
    xyoffset = -0.5
    angle = np.pi / 4

    rgc_rc = RadialGradientConvergence().run(
        gxmag,
        gymag,
        imgmag,
        doIntensityWeighting=False,
        magnification=mag,
        grad_magnification=grad_mag,
        radius=radius,
        offset=offset,
        xyoffset=xyoffset,
        angle=angle,
        sensitivity=1,
        run_type="opencl",
    )

    rgc_rc_iw = RadialGradientConvergence().run(
        gxmag,
        gymag,
        imgmag,
        doIntensityWeighting=True,
        magnification=mag,
        grad_magnification=grad_mag,
        radius=radius,
        offset=offset,
        xyoffset=xyoffset,
        angle=angle,
        sensitivity=1,
        run_type="opencl",
    )

    fig, ax = plt.subplots(1, 6)
    ax[0].imshow(np.mean(imgmag, axis=0), cmap="gray")
    ax[0].set_title("Image mag")
    ax[1].imshow(np.mean(rgc_rc, axis=0), cmap="gray")
    ax[1].set_title("RGC rc")
    ax[2].imshow(np.mean(rgc_rc_iw, axis=0), cmap="gray")
    ax[2].set_title("RGC rc iw")
    ax[3].imshow(np.mean(rgc_fd, axis=0), cmap="gray")
    ax[3].set_title("RGC fd")
    ax[4].imshow(np.mean(rgc_fd_iw, axis=0), cmap="gray")
    ax[4].set_title("RGC fd iw")
    ax[5].imshow(np.mean(rgc_rc - rgc_fd, axis=0), cmap="gray")
    ax[5].set_title("RGC fd - RGC fd")
    plt.show()
