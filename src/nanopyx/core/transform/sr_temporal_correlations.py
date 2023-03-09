import numpy as np


class TemporalCorrelation:
    accepted_correlation_types = ('mean', 'var', 'tac2')

    def __init__(self, correlation_type: str):
        """
        Perform a temporal correlation analysis of an image (with shape time, height, width) #TODO: discuss what shape should be the input
        :param correlation_type: desired type of interpolation ("mean", "var" or "tac2")
        """
        self.correlation_type = correlation_type


    def calculate_tc(self, im: np.array):

        out_array = np.empty((im.shape[0]))

        # assert isinstance(im, np.ndarray)

        if im.dtype != np.float32:
            im = np.array(im,dtype='float32')

        assert im.ndim == 3 #TODO: discuss if we should consider (t,c,z,r,c) here

        if self.correlation_type == "mean":
            out_array = np.mean(im, axis=0)
        
        elif self.correlation_type == "var":
            out_array = np.var(im, axis=0)
        
        elif self.correlation_type == "tac2": # second order autocorrelation function
                mean = np.mean(im, axis=0)  
                centered = im - mean  # center data around the mean
                nlag = 1  # number of lags to compute TAC2 for
                out_array = np.mean(centered[:-nlag] * centered[nlag:], axis=0)
        
        else:
             raise ValueError(f"Type of correlation must be one of {self.accepted_correlation_types}")

        return out_array

            





