# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

from libc.math cimport sqrt, fabs, exp, isnan, floor

from ..transform.interpolation_catmull_rom cimport _interpolate, Interpolator
from ..transform.interpolation_fft_zoom import magnify as fft_zoom
from ..transform.image_magnify import cv2_zoom as zoom
from ..utils.timeit import timeit2


import numpy as np
cimport numpy as np

from cython.parallel import prange

cdef float Gx_Gy_MAGNIFICATION = 2.0

cdef extern from "_c_gradients.h":
    void _c_gradient_3d(float* image, float* imGx, float* imGy, float* imGz, int d, int h, int w) nogil

cdef class RadialGradientConvergence3D:

    cdef int magnification
    cdef float fwhm
    cdef float sensitivity
    cdef float tSS # two sigma squared
    cdef float tSO # two sigma plus one
    cdef bint doIntensityWeighting

    def __init__(self, magnification: int = 5, radius: float = 1.5, sensitivity: float = 1 , doIntensityWeighting: bool = True):
        """
         Calculate the Radial Gradient Convergence (RGC) of an image.
         :param magnification: magnification of the image
         :param radius: radius of the RGC (the PSF Full-Width-Half-Maximum)
         :param sensitivity: sensitivity of the RGC (sharpening factor)
         :param doIntensityWeighting: whether to do intensity weighting
         """
        self.magnification = magnification
        self.fwhm = radius
        self.sensitivity = sensitivity
        self.doIntensityWeighting = doIntensityWeighting
 
        cdef float sigma = radius / 2.355
        self.tSS = 2 * sigma * sigma
        self.tSO = 2 * sigma + 1

    @timeit2
    def calculate(self, im: np.ndarray):
        """
        Calculate the RGC of an image-stack.
        :param im: the image to calculate the RGC of
        :return: the RGC of the image
        """
        assert im.ndim == 4

        nFrames = im.shape[0]

        cdef float [:,:,:,:] imRaw = im.astype(np.float32, copy=False)
        cdef float [:,:,:,:] imRad = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification, im.shape[3]*self.magnification), dtype=np.float32)
        cdef float [:,:,:,:] imInt = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification, im.shape[3]*self.magnification), dtype=np.float32) # interpolated image
        cdef float [:,:,:,:] imGx = np.zeros_like(imRaw) # Gradient of the interpolated image
        cdef float [:,:,:,:] imGy = np.zeros_like(imRaw)
        cdef float [:,:,:,:] imGz = np.zeros_like(imRaw)

        cdef int n

        with nogil:
            for n in prange(nFrames):
                _c_gradient_3d(&imRaw[n,0,0,0], &imGx[n,0,0,0], &imGy[n,0,0,0], &imGz[n,0,0,0], imRaw.shape[1], imRaw.shape[2], imRaw.shape[3])

        cdef int p
        for p in range(nFrames):
            self._single_frame_RGC_map(imRaw[p,:,:,:], imRad[p,:,:,:], imInt[p,:,:,:], imGx[p,:,:,:], imGy[p,:,:,:], imGz[p,:,:,:])

        return imRad, imInt, imGx, imGy, imGz
    
    cdef void _single_frame_RGC_map(self, float[:,:,:] imRaw, float[:,:,:] imRad, float[:,:,:] imInt, float[:,:,:] imGx, float[:,:,:] imGy, float[:,:,:] imGz):
        """
        Calculate the RGC map of an image frame.
        :param imRaw: the frame to calculate the RGC map of
        :param imRad: the RGC map of the frame (previously initialised as a 2D array the size of imRaw x magnification)
        :param imInt: the interpolated image (previously initialised as a 2D array the size of imRaw x magnification)
        :param imGx: the intensity gradients of the interpolated image in the horizontal direction (same size as imInt)
        :param imGy: the intensity gradients of the interpolated image in the vertical direction (same size as imInt)
        :return: the RGC of the image frame.
        """

        cdef int w = imRaw.shape[2]
        cdef int h = imRaw.shape[1]
        cdef int d = imRaw.shape[0]
        cdef int magnification = self.magnification
        cdef int yM, xM

        #TODO: interpolate 3D image
        # cdef Interpolator interpolator = Interpolator(imRaw)
        # imInt[:,:] = interpolator._magnify(self.magnification)

        # self._calculate_3d_gradient(imRaw, imGx, imGy, imGz) # calculate gradients of the interpolated image

        # with nogil:
        for zM in range(magnification, d * magnification):
            for yM in range(magnification, h * magnification):
                for xM in range(magnification, w * magnification):
                    if self.doIntensityWeighting:
                        imRad[zM, yM, xM] = self._calculate_3d_RGC(xM, yM, zM, imGx, imGy, imGz) * imInt[zM, yM, xM]
                    else:
                        imRad[zM, yM, xM] = self._calculate_3d_RGC(xM, yM, zM, imGx, imGy, imGz)

        
    cdef void _calculate_3d_gradient(self, float[:,:,:] image, float[:,:,:] imGx, float[:,:,:] imGy, float[:,:,:] imGz):
        cdef int w = image.shape[2]
        cdef int h = image.shape[1]
        cdef int d = image.shape[0]

        cdef float ip0, ip1, ip2, ip3, ip4, ip5, ip6, ip7

        cdef int z_i, y_i, x_i

        for z_i in range(d-1):
            for y_i in range(h-1):
                for x_i in range(w-1):
                    ip0 = image[z_i, y_i, x_i]
                    ip1 = image[z_i, y_i, x_i + 1]
                    ip2 = image[z_i, y_i + 1, x_i]
                    ip3 = image[z_i, y_i + 1, x_i + 1]
                    ip4 = image[z_i + 1, y_i, x_i]
                    ip5 = image[z_i + 1, y_i, x_i + 1]
                    ip6 = image[z_i + 1, y_i + 1, x_i]
                    ip7 = image[z_i + 1, y_i + 1, x_i + 1]
                    imGx[z_i, y_i, x_i] = (ip1 + ip3 + ip5 + ip7 - ip0 - ip2 - ip4 - ip6) / 4
                    imGy[z_i, y_i, x_i] = (ip2 + ip3 + ip6 + ip7 - ip0 - ip1 - ip4 - ip5) / 4
                    imGz[z_i, y_i, x_i] = (ip4 + ip5 + ip6 + ip7 - ip0 - ip1 - ip2 - ip3) / 4

    
    cdef float _calculate_3d_RGC(self, int xM, int yM, int zM, float[:,:,:] imGx, float[:,:,:] imGy, float[:,:,:] imGz):
        cdef int w = imGx.shape[2]
        cdef int h = imGx.shape[1]
        cdef int d = imGx.shape[0]

        cdef float vx, vy, vz, Gx, Gy, Gz

        cdef float dx, dy, dz
        cdef float distance, distanceWeight, GdotR, Dk

        cdef float xc = (xM + 0.5) / self.magnification # subpixel in the centre
        cdef float yc = (yM + 0.5) / self.magnification # subpixel in the centre
        cdef float zc = (zM + 0.5) / self.magnification

        cdef float RGC = 0 # Radial Gradient Convergence
        cdef float distanceWeightSum = 0

        cdef int _start = -(<int>(Gx_Gy_MAGNIFICATION * self.fwhm))
        cdef int _end = <int>(Gx_Gy_MAGNIFICATION * self.fwhm + 1)

        cdef int i, j, k

        for k in range(_start, _end):
            vz = (<int>(Gx_Gy_MAGNIFICATION * zc) + k) / Gx_Gy_MAGNIFICATION

            if 0 < vz <= d - 1:
                for j in range(_start, _end):
                    vy = (<int>(Gx_Gy_MAGNIFICATION * yc) + j) / Gx_Gy_MAGNIFICATION # position in continuous space

                    if 0 < vy <= h - 1:

                        for i in range(_start, _end):
                            vx = (<int>(Gx_Gy_MAGNIFICATION * xc) + i) / Gx_Gy_MAGNIFICATION # position in continuous space

                            if 0 < vx <= w - 1:

                                dx = vx - xc
                                dy = vy - yc
                                dz = vz - zc
                                distance = sqrt(dx * dx + dy * dy + dz * dz)

                                if distance != 0 and distance <= self.tSO:
                                    #TODO: interpolate gradients in 3D
                                    Gx = 1
                                    Gy = 1
                                    Gz = 1

                                    distanceWeight = self._calculateDW(distance)
                                    distanceWeightSum += distanceWeight
                                    GdotR = Gx*dx + Gy*dy + Gz*dz

                                    if GdotR < 0:
                                        Dk = self._calculateDk(Gx, Gy, Gz, dx, dy, dz, distance)
                                        RGC += Dk * distanceWeight
        RGC /= distanceWeightSum
        if RGC >= 0:
            RGC = RGC ** self.sensitivity
        else:
            RGC = 0

        return RGC


    cdef float _calculateDW(self, float distance) nogil: # distance weight
        return (distance * exp((-distance * distance) / self.tSS)) ** 4

    
    cdef float _calculateDk(self, float Gx, float Gy, float Gz, float dx, float dy, float dz, float distance) nogil:
        Dk = fabs(Gy * dz - Gz * dy - Gx * dz + Gz * dx + Gx * dy - Gy * dx) / sqrt(Gx * Gx + Gy * Gy + Gz * Gz)
        if isnan(Dk):
            Dk = distance
        Dk = 1 - Dk / distance # if 1: vector pointing exactly to the centre
        return Dk