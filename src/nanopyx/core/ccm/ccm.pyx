cimport numpy as np

def calculate_ccm(np.ndarray img_stack, int ref):
    return _calculate_ccm(img_stack, ref)

cdef float[:, :, :] _calculate_ccm(float[:, :, :] img_stack, int ref):
    pass