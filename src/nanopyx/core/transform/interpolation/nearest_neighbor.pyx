# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True, autogen_pxd=True

import numpy as np
cimport numpy as np

from libc.math cimport pi, hypot, cos, sin, atan2, log, exp

from cython.parallel import prange

# nearest-neighbor interpolation of a 2D array
cdef double _interpolate(float[:,:] image, double x, double y) nogil:
    """
    Interpolate a 2D array using nearest-neighbor interpolation
    :param image: 2D array to interpolate
    :param x: x position to interpolate
    :param y: y position to interpolate
    :return: interpolated value
    """

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]

    # return 0 if x OR y positions do not exist in image
    if not 0 <= x < w or not 0 <= y < h:
        return 0

    cdef int x0 = int(x)
    cdef int y0 = int(y)    

    return image[y0, x0]


cdef class Interpolator:

    # autogen_pxd: cdef float[:,:] image
    # autogen_pxd: cdef int w, h
    # autogen_pxd: cdef object original_dtype

    def __init__(self, image):
        """
        Interpolate a 2D array
        :param image: 2D array to interpolate
        """

        assert image.ndim == 2, "image must be 2D"
        
        if type(image) is np.ndarray:
            self.image = image.view(np.float32)
            self.original_dtype = image.dtype
        else: # assume its a memoryview
            self.image = image
            self.original_dtype = np.float32

        self.w = image.shape[1]
        self.h = image.shape[0]


    cdef float _interpolate(self, float x, float y) nogil:
        """
        Interpolate a 2D array using nearest-neighbor interpolation
        :param x: x position to interpolate
        :param y: y position to interpolate
        :return: interpolated value
        """
        return _interpolate(self.image, x, y)

    def magnify(self, int magnification) -> np.ndarray:
        """
        Magnify an image by a factor of magnification
        :param magnification: magnification factor
        :return: magnified image
        """
        imMagnified = self._magnify(magnification)
        return np.asarray(imMagnified).astype(self.original_dtype)


    cdef float[:,:] _magnify(self, float magnification):
        cdef int i, j
        cdef int wM = int(self.w * magnification)
        cdef int hM = int(self.h * magnification)
        cdef float x, y

        cdef float[:,:] imMagnified = np.empty((hM, wM), dtype=np.float32)

        with nogil:
            for i in prange(wM):
                x = i / magnification
                for j in range(hM):
                    y = j / magnification
                    imMagnified[j, i] = self._interpolate(x, y)
        
        return imMagnified

    def magnify_xy(self, int magnification_y, int magnification_x) -> np.ndarray:
        """
        Magnify an image by a factor of magnification
        :param magnification: magnification factor
        :return: magnified image
        """
        imMagnified = self._magnify_xy(magnification_y, magnification_x)
        return np.asarray(imMagnified).astype(self.original_dtype)

    cdef float[:,:] _magnify_xy(self, float magnification_y, float magnification_x):
        cdef int i, j
        cdef int wM = int(self.w * magnification_x)
        cdef int hM = int(self.h * magnification_y)
        cdef float x, y

        cdef float[:,:] imMagnified = np.empty((hM, wM), dtype=np.float32)

        with nogil:
            for i in prange(wM):
                x = i / magnification_x
                for j in range(hM):
                    y = j / magnification_y
                    imMagnified[j, i] = self._interpolate(x, y)
        
        return imMagnified

    def scale_xy(self, float scaling_y, float scaling_x) -> np.ndarray:
        """
        Scale an image by a factor. 
        Equivalent to magnify_xy but takes floats as inputs and mantains image shape.
        :param scaling_y: scale factor in y
        :param scaling_x: scale factor in x
        :return: scaled image
        """
        imScaled = self._scale_xy(scaling_y, scaling_x)
        return np.asarray(imScaled).astype(self.original_dtype)

    cdef float[:,:] _scale_xy(self, float scaling_y, float scaling_x):
        cdef int i, j
        cdef float x, y
        cdef float wM = self.w * scaling_x
        cdef float hM = self.h * scaling_y
        cdef float[:,:] imScaled = np.empty((self.h, self.w), dtype=np.float32)

        with nogil:
            for i in prange(self.w):
                x = (i+(wM/2-self.w/2)) / scaling_x
                for j in range(self.h):
                    y = (j+(hM/2-self.h/2)) / scaling_y
                    imScaled[j, i] = self._interpolate(x, y)
        
        return imScaled

    def shift(self, double dx, double dy) -> np.ndarray:
        """
        Shift an image by (dx, dy) using interpolation
        :param dx: shift along x-axis
        :param dy: shift along y-axis
        :return: shifted image
        """
        imShifted = self._shift(dx, dy)
        return np.asarray(imShifted).astype(self.original_dtype)


    cdef float[:,:] _shift(self, float dx, float dy):
        
        cdef float[:,:] imShifted = np.zeros((self.h, self.w), dtype=np.float32)
        
        cdef int i, j
        cdef int _dx = int(dx)
        cdef int _dy = int(dy)
        cdef int x_start = max(0, _dx)
        cdef int y_start = max(0, _dy)
        cdef int x_end = min(self.w, self.w + _dx)
        cdef int y_end = min(self.h, self.h + _dy)
        
        with nogil:
            for i in prange(x_start, x_end):
                for j in range(y_start, y_end):
                    imShifted[j,i] = self._interpolate(i - dx, j - dy)

        return imShifted 


    def rotate(self, float angle, float cx=-1, float cy=-1) -> np.ndarray:
        """
        Rotate an image by angle radians around (cx,cy) using interpolation
        :param angle: rotation angle in radians, positive angles are counter clockwise
        :param cx: x coordinate of the center of rotation, defaults to image center if negative
        :param cy: y coordinate of the center of rotation, defaults to image center if negative
        """
        if cx<0 or cy<0:
            cx = self.w / 2
            cy = self.h / 2

        imRotated = self._rotate(angle, cx, cy)
        return np.asarray(imRotated).astype(self.original_dtype)


    cdef float[:,:] _rotate(self, float angle, float cx, float cy):

        cdef float[:,:] imRotated = np.zeros((self.h, self.w), dtype=np.float32)
        
        cdef int i, j
        cdef float rotx, roty

        cdef float cosine = cos(angle)
        cdef float sine = sin(angle)

        with nogil:
            for i in prange(0, self.w):
                for j in range(0, self.h):
                    rotx = cosine*(i-cx) - sine*(j-cy) + cx
                    roty = sine*(i-cx) + cosine*(j-cy) + cy
                    imRotated[j,i] = self._interpolate(rotx,roty)

        return imRotated

    def polar(self, str scale='linear') -> np.ndarray:
        """
        Transforms an image into its polar coordinate equivalent with origin at the center of the image
        :param scale: scaling done during conversion, if 'log' performs log-polar transformation
        :return: (theta,r) image array 
        """
        polarized = self._polar(scale)
        return np.asarray(polarized).astype(self.original_dtype)

    cdef float[:,:] _polar(self, str scale):

        cdef float cx = self.w / 2
        cdef float cy = self.h / 2

        cdef int max_theta = 360
        cdef int max_radius = int(hypot(cx,cy))+1
        
        cdef float[:,:] polarized = np.zeros((max_theta, max_radius), dtype=np.float32)

        cdef int i,j
        cdef float x,y

        with nogil:
            for i in prange(0,max_radius):
                for j in range(0,max_theta):
                    if scale=='log':
                        x = exp(i*log(max_radius)/max_radius) * cos(j*pi/180) + cx
                        y = exp(i*log(max_radius)/max_radius) * sin(j*pi/180) + cy
                    else:
                        x = i * cos(j*pi/180) + cx
                        y = i * sin(j*pi/180) + cy
                    polarized[j,i] = self._interpolate(x,y)

        return polarized

    def cartesian(self, int x_shape, int y_shape, str scale='linear')-> np.ndarray:
        """
        Transforms an image into its cartesian coordinate equivalent. Assumes image shape is (theta,r) and the origin is at the center of the cartesian image
        :param x_shape: width of original image
        :param y_shape: height of original image
        :param scale: scaling performed during transition to polar coordinates, if 'log' assumes the image was a log-polar image
        :return: (y,x) image array
        """
        cart = self._cartesian(x_shape, y_shape, scale)
        return np.asarray(cart).astype(self.original_dtype)

    cdef float[:,:] _cartesian(self, int x_shape, int y_shape, str scale):
        
        cdef float[:,:] cart = np.zeros((y_shape, x_shape), dtype=np.float32)

        cdef float cx = x_shape / 2
        cdef float cy = y_shape / 2

        cdef int i, j
        cdef float r, t
        with nogil:
            for i in prange(0,x_shape):
                for j in range(0,y_shape):
                    if scale=='log' and (j!=cy and i!=cx):
                        r = log(hypot(j-cy,i-cx)) * self.w / log(self.w)
                    else:
                        r = hypot(j-cy, i-cx)
                    t = atan2(j-cy,i-cx) * 180/pi
                    if t<0:
                        t = 360+t
                    cart[j,i] = self._interpolate(r,t)

        return cart 