# Code below is autogenerated by pyx2pxd - https://github.com/HenriquesLab/pyx2pxd

cdef float[:, :, :] _normalizeFFT(float[:, :] fft_real, float[:, :] fft_imag)
cdef float[:, :] _apodize_edges(float[:, :] img)
cdef double _linmap(float val, float valmin, float valmax, float mapmin, float mapmax)
cdef float[:, :] _get_mask(int w, float r2)
cdef double _get_corr_coef_norm(float[:, :] fft_real, float[:, :] fft_imag, float[:, :] mask)
cdef double[:] _get_max(float[:] arr, int x1, int x2)
cdef double[:] _get_min(float[:] arr, int x1, int x2)
cdef _get_best_score(float[:] kc, float[:] a)
cdef _get_max_score(float[:] kc, float[:] a)
