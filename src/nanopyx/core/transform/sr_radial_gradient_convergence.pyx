# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=False

from libc.math cimport sqrt, fabs, exp, isnan, floor, pow

from ..transform.interpolation_catmull_rom cimport _interpolate, Interpolator
from ..transform.image_magnify import cv2_zoom as zoom
# from ..transform.image_magnify import fourier_zoom as zoom
from ..utils.timeit import timeit2
from ..transform.interpolation_fft_zoom import magnify as fft_zoom
from nanopyx.liquid import CRShiftAndMagnify


import numpy as np
cimport numpy as np

from cython.parallel import prange

cdef float Gx_Gy_MAGNIFICATION = 2.0

cdef class RadialGradientConvergence:

    # autogen_pxd: cdef int magnification
    # autogen_pxd: cdef float fwhm
    # autogen_pxd: cdef float sensitivity
    # autogen_pxd: cdef float tSS # two sigma squared
    # autogen_pxd: cdef float tSO # two sigma plus one
    # autogen_pxd: cdef bint doIntensityWeighting


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
        assert im.ndim == 3

        nFrames = im.shape[0]

        cdef float [:,:,:] imRaw = im.astype(np.float32, copy=False)

        # Interpolate the image stack for intensity weighting
        crsm = CRShiftAndMagnify()
        cdef float [:,:,:] imInt = crsm.run(imRaw, 0, 0, self.magnification, self.magnification)

        # Calculate intensity gradients of the Raw image
        cdef float [:,:,:] imGx = np.zeros_like(imRaw) 
        cdef float [:,:,:] imGy = np.zeros_like(imRaw)

        cdef int n
        with nogil: # will change this soon (to go under single_frame_RGC_map)
            for n in prange(nFrames):
                _c_gradient_roberts_cross(&imRaw[n,0,0], &imGx[n,0,0], &imGy[n,0,0], imRaw.shape[1], imRaw.shape[2])
        
        # Interpolate the Gradients
        cdef float [:,:,:] imIntGx = crsm.run(imGx, 0, 0, self.magnification*Gx_Gy_MAGNIFICATION, self.magnification*Gx_Gy_MAGNIFICATION)
        cdef float [:,:,:] imIntGy = crsm.run(imGy, 0, 0, self.magnification*Gx_Gy_MAGNIFICATION, self.magnification*Gx_Gy_MAGNIFICATION)
    
        cdef float [:,:,:] imRad = np.zeros((im.shape[0], im.shape[1]*self.magnification, im.shape[2]*self.magnification), dtype=np.float32)

        cdef int p
        for p in range(nFrames):
            self._single_frame_RGC_map(imRaw[p,:,:], imRad[p,:,:], imInt[p,:,:], imIntGx[p,:,:], imIntGy[p,:,:])

        return imRad, imInt, imIntGx, imIntGy


    cdef void _single_frame_RGC_map(self, float[:,:] imRaw, float[:,:] imRad, float[:,:] imInt, float[:,:] imIntGx, float[:,:] imIntGy):
        cdef int w = imRaw.shape[1]
        cdef int h = imRaw.shape[0]
        cdef int yM, xM

        with nogil:
            for yM in prange(self.magnification, h * self.magnification):
                for xM in range(self.magnification, w * self.magnification):
                    if self.doIntensityWeighting:
                        imRad[yM, xM] = self._calculateRGC(xM, yM, imIntGx, imIntGy, imInt) * imInt[yM, xM] 
                    else:
                        imRad[yM, xM] = self._calculateRGC(xM, yM, imIntGx, imIntGy, imInt)


    cdef float _calculateRGC(self, int xM, int yM, float[:,:] imIntGx, float[:,:] imIntGy, float[:,:] imInt) nogil:

        cdef int w = imInt.shape[1]
        cdef int h = imInt.shape[0]

        cdef float vx, vy, Gx, Gy

        cdef float dx, dy
        cdef float distance, distanceWeight, GdotR, Dk

        cdef float xc = (xM + 0.5) / self.magnification # subpixel in the centre
        cdef float yc = (yM + 0.5) / self.magnification # subpixel in the centre

        cdef float RGC = 0 # Radial Gradient Convergence
        cdef float distanceWeightSum = 0

        cdef int _start = -(<int>(Gx_Gy_MAGNIFICATION * self.fwhm))
        cdef int _end = <int>(Gx_Gy_MAGNIFICATION * self.fwhm + 1)

        cdef int i, j

        for j in range(_start, _end): 
            vy = (<int>(Gx_Gy_MAGNIFICATION * yc) + j) / Gx_Gy_MAGNIFICATION # position in continuous space

            if 0 < vy <= h - 1:

                for i in range(_start, _end):
                    vx = (<int>(Gx_Gy_MAGNIFICATION * xc) + i) / Gx_Gy_MAGNIFICATION # position in continuous space

                    if 0 < vx <= w - 1:

                        dx = vx - xc
                        dy = vy - yc
                        distance = sqrt(dx * dx + dy * dy)

                        if distance != 0 and distance <= self.tSO:

                            Gx = imIntGx[<int>((vy)*self.magnification*Gx_Gy_MAGNIFICATION), <int>((vx)*self.magnification*Gx_Gy_MAGNIFICATION)]
                            Gy = imIntGy[<int>((vy)*self.magnification*Gx_Gy_MAGNIFICATION), <int>((vx)*self.magnification*Gx_Gy_MAGNIFICATION)]

                            #distanceWeight = self._calculateDW(distance)
                            distanceWeight = _c_calculate_dw(distance, self.tSS)
                            distanceWeightSum += distanceWeight
                            GdotR = Gx*dx + Gy*dy

                            if GdotR < 0: # if the vector is pointing inwards
                                Dk = _c_calculate_dk(Gx, Gy, dx, dy, distance)
                                # Dk = self._calculateDk(Gx, Gy, dx, dy, distance)
                                RGC += Dk * distanceWeight

        RGC /= distanceWeightSum

        if RGC >= 0 and self.sensitivity > 1:
            RGC = RGC ** self.sensitivity
        elif RGC < 0:
            RGC = 0

        return RGC
