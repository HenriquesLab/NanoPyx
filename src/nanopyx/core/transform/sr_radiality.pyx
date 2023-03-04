# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=False, autogen_pxd=True

"""
Python reimplementation of the Radiality Transform from the original SRRF paper
Paper: https://www.nature.com/articles/ncomms12471
Original code: https://github.com/HenriquesLab/NanoJ-SRRF/blob/master/SRRF/src/nanoj/srrf/java/SRRF.java
"""

from libc.math cimport sqrt, pi, fabs, cos, sin
from libc.stdlib cimport free

import numpy as np
cimport numpy as np

from .interpolation_catmull_rom cimport _interpolate

from cython.parallel import prange

cdef class Radiality:
    # autogen_pxd: cdef int magnification, symmetryAxis, border, nRingCoordinates
    # autogen_pxd: cdef float ringRadius, psfWidth, gradRadius
    # autogen_pxd: cdef bint doIntegrateLagTimes, radialityPositivityConstraint, doIntensityWeighting
    # autogen_pxd: cdef float[12] xRingCoordinates, yRingCoordinates

    def __init__(self, magnification: int = 5, ringRadius: float = 0.5, border: int = 0, radialityPositivityConstraint: bool = True, doIntensityWeighting: bool = True):
        """
        Calculate Radiality, as defined on the original version of SRRF - REF: https://www.nature.com/articles/ncomms12471
        :param magnification: Desired magnification for the generated radiality image
        :param ringRadius: Radius of the ring used to calculate the radiality
        :param border: Number of pixels to be zeroed on the borders of the radiality image
        :param radialityPositivityConstraint: If True, the radiality image will be constrained to be positive (values >= 0)
        :param doIntensityWeighting: If True, the radiality image will be weighted by the intensity of the original image
        """
        self.magnification = magnification
        self.border = border
        self.ringRadius = ringRadius * magnification
        self.border = border
        self.radialityPositivityConstraint = radialityPositivityConstraint
        self.doIntensityWeighting = doIntensityWeighting
        self.nRingCoordinates = 12

        cdef float angleStep = (pi * 2.) / self.nRingCoordinates
        with nogil:
            for angleIter in range(self.nRingCoordinates):
                self.xRingCoordinates[angleIter] = cos(angleStep * angleIter) * self.ringRadius
                self.yRingCoordinates[angleIter] = sin(angleStep * angleIter) * self.ringRadius

    def __dealloc__(self):
        # if self.xRingCoordinates is not NULL:
        #     free(self.xRingCoordinates)
        #     free(self.yRingCoordinates)
        pass

    def calculate(self, image_stack: np.ndarray):
        """
        Calculate Radiality, as defined on the original version of SRRF - REF: https://www.nature.com/articles/ncomms12471
        :param image_stack: Image stack to be processed
        :return: Radiality image-stack, magnified image-stack, x-gradient image, y-gradient image
        """
        assert image_stack.ndim == 3

        cdef int nFrames = image_stack.shape[0]

        imRaw = image_stack.astype(np.float32)
        imGx = np.zeros_like(imRaw)
        imGy = np.zeros_like(imRaw)
        imRad = np.zeros((image_stack.shape[0], image_stack.shape[1]*self.magnification, image_stack.shape[2]*self.magnification), dtype=np.float32)
        imIW = np.zeros((image_stack.shape[0], image_stack.shape[1]*self.magnification, image_stack.shape[2]*self.magnification), dtype=np.float32)

        cdef int n
        cdef float[:,:,:] _imRaw = imRaw
        cdef float[:,:,:] _imGx = imGx
        cdef float[:,:,:] _imGy = imGy
        cdef float[:,:,:] _imRad = imRad
        cdef float[:,:,:] _imIW = imIW

        with nogil:
            for n in prange(nFrames): #, schedule='static', chunksize=1):
                self._calculate_radiality(_imRaw[n,:,:], _imRad[n,:,:], _imIW[n,:,:], _imGx[n,:,:], _imGy[n,:,:], 0, 0)

        return imRad, imIW, imGx, imGy

    cdef void _calculate_radiality(self, float[:,:] imRaw, float[:,:] imRad, float[:,:] imIW, float[:,:] imGx, float[:,:] imGy, float shiftX, float shiftY) nogil:
        """
        Note that Gx and Gy are initialized but zeroed
        """

        cdef int w = imRaw.shape[1]
        cdef int h = imRaw.shape[0]
        cdef int i, j, sampleIter
        cdef float x0, y0, xc, yc, GMag, xRing, yRing

        # calculate Gx and Gy
        cdef float vGx, vGy
        cdef float CGH # for Culley Gustafsson Henriques transform

        # Radiality Variable
        cdef float Dk, DivDFactor = 0

        for j in range(1, h-1):
            for i in range(1, w-1):
                imGx[j,i] = -imRaw[j,i-1]+imRaw[j,i+1]
                imGy[j,i] = -imRaw[j-1,i]+imRaw[j+1,i]

        for j in range((1 + self.border) * self.magnification, (h - 1 - self.border) * self.magnification):
            for i in range((1 + self.border) * self.magnification, (w - 1 - self.border) * self.magnification):
                xc = i + 0.5 + shiftX * self.magnification
                yc = j + 0.5 + shiftY * self.magnification

                imIW[j,i] = _interpolate(imRaw, xc / self.magnification, yc / self.magnification)

                # Output
                CGH = 0
                for sampleIter in range(0, self.nRingCoordinates):
                    xRing = self.xRingCoordinates[sampleIter]
                    yRing = self.yRingCoordinates[sampleIter]

                    x0 = xc + xRing
                    y0 = yc + yRing

                    vGx = _interpolate(imGx, x0 / self.magnification, y0 / self.magnification)
                    vGy = _interpolate(imGy, x0 / self.magnification, y0 / self.magnification)
                    GMag = sqrt(vGx * vGx + vGy * vGy)

                    Dk = 1 - self._calculateDk(x0, y0, xc, yc, vGx, vGy, GMag) / self.ringRadius
                    Dk = Dk * Dk

                    if (vGx * xRing + vGy * yRing) > 0: # inwards or outwards vector
                        DivDFactor -= Dk
                    else:
                        DivDFactor += Dk

                DivDFactor /= self.nRingCoordinates

                if self.radialityPositivityConstraint:
                    CGH = max(DivDFactor, 0)
                else:
                    CGH = DivDFactor

                if self.doIntensityWeighting:
                    imRad[j,i] = CGH

                else:
                    imRad[j,i] = CGH * imIW[j,i]


    cdef float _calculateDk(self, float x, float y, float xc, float yc, float vGx, float vGy, float vGx2Gy2) nogil:
        if vGx2Gy2 == 0:
            return self.ringRadius
        else:
            return fabs(vGy * (xc - x) - vGx * (yc - y)) / vGx2Gy2
