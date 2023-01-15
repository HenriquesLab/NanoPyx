# cython: infer_types=True, wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, profile=True
# nanopyx: autogen_pxd=False

import numpy as np
cimport numpy as np

from cython.parallel import prange

# nearest-neighbor interpolation of a 2D array
cdef double _interpolate(float[:,:] image, double x, double y) nogil:
    """
    Interpolate a 2D array using nearest-neighbor interpolation.

    Parameters
    ----------
    image : 2D array
        The image to interpolate.
    x : float
        The x coordinate to interpolate at.
    y : float
        The y coordinate to interpolate at.
    
    Returns
    -------
    float
        The interpolated value.
    """

    cdef int w = image.shape[1]
    cdef int h = image.shape[0]

    # return 0 if x and y positions do not exist in image
    if not 0 <= x < w and not 0 <= y < h:
        return 0

    cdef int x0 = int(x)
    cdef int y0 = int(y)    

    return image[y0, x0]


cdef class Interpolator:

    def __init__(self, image):
        """
        Interpolate a 2D array

        Parameters
        ----------
        image : 2D array
            The image to interpolate.
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
        return _interpolate(self.image, x, y)


    def magnify(self, int magnification) -> np.ndarray:
        """
        Magnify an image by a factor of magnification.

        Parameters:
            im (np.ndarray): image to magnify.
            magnification (int): magnification factor.

        Returns:
            Magnified image (np.ndarray).
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


    def shift(self, double dx, double dy) -> np.ndarray:
        """
        Shift an image by (dx, dy) using interpolation.

        Parameters:
            im (np.ndarray): image to shift.
            dx (float): shift along x-axis.
            dy (float): shift along y-axis.

        Returns:
            Shifted image (np.ndarray).
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