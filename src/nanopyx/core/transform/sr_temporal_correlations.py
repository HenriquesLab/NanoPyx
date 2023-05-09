import numpy as np


def calculate_SRRF_temporal_correlations(im: np.array, order: int = 1, do_integrate_lag_times: bool = 0): #these are for SRRF

    im = np.array(im, dtype='float32')
    assert im.ndim == 3 and order <= 4
    
    if order == 0: #TODO: check order number in NanoJ
        out_array = np.amax(im, axis=0)
    
    elif order == 1:
        out_array = np.mean(im, axis=0)
    
    elif order == -1:
        out_array = calculate_pairwise_product_sum(im)

    else: # order = 2 or order = 3 or order = 4
        out_array = calculate_acrf_(im, order, do_integrate_lag_times)
    
    return out_array


# def calculate_eSRRF_temporal_correlations()

    
def calculate_pairwise_product_sum(rad_array):
    n_time_points, height_m, width_m = rad_array.shape

    out_array = np.zeros((height_m, width_m), dtype=np.float32)
    #max_array = np.max(rad_array, axis=0)
    counter = 0
    pps = 0

    for t0 in range(n_time_points):
        r0 = np.maximum(rad_array[t0],0)
        if np.any(r0) > 0:
            for t1 in range(t0, n_time_points):
                r1 = np.maximum(rad_array[t1],0)
                pps += r0 * r1
                counter += 1
        else:
            counter += n_time_points - t0
    pps = pps/ max(counter,1)
    out_array = pps

    return out_array


def calculate_acrf_(rad_array, order, do_integrate_lag_times):
    im = rad_array.copy()
    n_time_points, height_m, width_m = im.shape
    mean = np.mean(im, axis=0)

    out_array = np.zeros((height_m, width_m), dtype=np.float32)

    abcd = np.zeros((height_m, width_m), dtype=np.float32)
    abc = np.zeros((height_m, width_m), dtype=np.float32)
    ab = np.zeros((height_m, width_m), dtype=np.float32)
    cd = np.zeros((height_m, width_m), dtype=np.float32)
    ac = np.zeros((height_m, width_m), dtype=np.float32)
    bd = np.zeros((height_m, width_m), dtype=np.float32)
    ad = np.zeros((height_m, width_m), dtype=np.float32)
    bc = np.zeros((height_m, width_m), dtype=np.float32)

    if do_integrate_lag_times != 1:
        t = 0
        while (t < n_time_points - order):
            ab = ab + (im[t] - mean) * (im[t+1] - mean)
            if order == 3:
                abc = abc + (im[t] - mean) * (im[t+1] - mean) + (im[t+2] - mean)
            if order == 4:
                a = im[t] - mean
                b = im[t+1] - mean
                c = im[t+2] - mean
                d = im[t+3] - mean
                abcd = abcd + np.multiply(np.multiply(np.multiply(a,b),c),d)
                cd = cd + np.multiply(c,d)
                ac = ac + np.multiply(a,c)
                bd = bd + np.multiply(b,d)
                ad = ad + np.multiply(a,d)
                bc = bc + np.multiply(b,c)
            t = t + 1
        if order == 3:
            out_array = np.absolute(abc) / n_time_points
        elif order == 4:
            out_array = np.absolute(abcd - ab * cd - ac * bd - ad * bc) / n_time_points
        else:
            out_array = np.absolute(ab) / n_time_points
    
    else:
        n_binned_time_points = n_time_points
        tbin = 0
        while n_binned_time_points > order:
            t = 0
            ab = np.zeros((height_m, width_m), dtype=np.float32)
            while (t < n_binned_time_points - order):
                tbin = t * order
                ab = ab + (im[t] - mean) * (im[t+1] - mean)
                if order == 3:
                    abc = abc + (im[t] - mean) * (im[t+1] - mean) + (im[t+2] - mean)
                if order == 4:
                    a = im[t] - mean
                    b = im[t+1] - mean
                    c = im[t+2] - mean
                    d = im[t+3] - mean
                    abcd = abcd + np.multiply(np.multiply(np.multiply(a,b),c),d)
                    cd = cd + np.multiply(c,d)
                    ac = ac + np.multiply(a,c)
                    bd = bd + np.multiply(b,d)
                    ad = ad + np.multiply(a,d)
                    bc = bc + np.multiply(b,c)
                    
                im[t] = np.zeros((height_m, width_m), dtype=np.float32)

                if tbin < n_binned_time_points:
                    for _t in range(order-1):
                        im[t] = im[t] + np.divide(im[tbin + _t], order)
                t = t + 1
            if order == 3:
                out_array = np.absolute(abc) / n_binned_time_points
            elif order == 4:
                out_array = np.absolute(abcd - ab * cd - ac * bd - ad * bc) / n_binned_time_points
            else:
                out_array = np.absolute(ab) / n_binned_time_points

            n_binned_time_points = n_binned_time_points / order

    return out_array


def calculate_tac2(rad_array):
    mean = np.mean(rad_array, axis=0)  
    centered = rad_array - mean  # center data around the mean
    nlag = 1  # number of lags to compute TAC2 for
    out_array = np.mean(centered[:-nlag] * centered[nlag:], axis=0)

    return out_array




# class TemporalCorrelation:
#     accepted_correlation_types = ('mean', 'var', 'tac2', 'mip', 'pair_wise_product_sum')

#     def __init__(self, correlation_type: str):
#         """
#         Perform a temporal correlation analysis of an image (with shape time, height, width) #TODO: discuss what shape should be the input
#         :param correlation_type: desired type of interpolation ("mean", "var" or "tac2")
#         """
#         self.correlation_type = correlation_type


#     def calculate_tc(self, im: np.array):

#         out_array = np.empty((im.shape[0]))

#         # assert isinstance(im, np.ndarray)

#         if im.dtype != np.float32:
#             im = np.array(im,dtype='float32')

#         assert im.ndim == 3 #TODO: discuss if we should consider (t,c,z,r,c) here

#         if self.correlation_type == "mean":
#             out_array = np.mean(im, axis=0)
        
#         elif self.correlation_type == "var":
#             out_array = np.var(im, axis=0)
        
#         elif self.correlation_type == "tac2": # second order autocorrelation function
#             mean = np.mean(im, axis=0)  
#             centered = im - mean  # center data around the mean
#             nlag = 1  # number of lags to compute TAC2 for
#             out_array = np.mean(centered[:-nlag] * centered[nlag:], axis=0)

#         elif self.correlation_type == "mip": # maximum intensity projection
#              out_array = np.amax(im, axis=0) #calculate_mip(im, doIntensityWeighting)

#         elif self.correlation_type == "pair_wise_product_sum":
#             out_array = calculate_pairwise_product_sum_(im)
        
#         else:
#              raise ValueError(f"Type of correlation must be one of {self.accepted_correlation_types}")

#         return out_array
    
    
    # def calculate_mip(im: np.array, doIntensityWeighting: bool = True):
    #     """
    #     Calculate Maximum Intensity Projection of an input image
    #     """
    #     #nb_time_points, height, width = im.shape
    #     out = np.empty((im.shape[0]))

    #     if doIntensityWeighting == True:
    #         out = 
    #     else:
    #         out = np.amax(im, axis=0)
    #     return out

    # def calculate_pairwise_product_sum(im: np.array):
    #     """
    #     Calculate Pair-Wise Product Sum of a 3D input image
    #     """



    # def calculate_pairwise_product_sum(self, rad_array: np.array):
    #     n_time_points, height_m, width_m = rad_array.shape

    #     SRRF_array = np.zeros((height_m, width_m), dtype=np.float32)

    #     r0 = np.maximum(rad_array[:, :, None, :], 0) 
    #     r1 = np.maximum(rad_array[:, :, :, None], 0) 

    #     pps = np.sum(r0 * r1, axis=(2, 3)) / np.maximum((n_time_points * (n_time_points + 1)) / 2, 1)
    #     SRRF_array = pps[0:height_m, 0:width_m]

    #     return SRRF_array


    # def calculate_pairwise_product_sum_(self, rad_array):
    #     n_time_points, height_m, width_m = rad_array.shape

    #     SRRF_array = np.zeros((height_m, width_m), dtype=np.float32)
    #     max_array = np.max(rad_array, axis=0)

    #     for i in range(height_m):
    #         for j in range(width_m):
    #             r0 = np.maximum(max_array[i][j], 0)  # maximum value at position (i,j)
    #             r1 = np.maximum(max_array, 0)  # maximum values of all pixels in the image
    #             pps = np.sum(r0 * r1) / np.maximum((n_time_points * (n_time_points + 1)) / 2, 1)
    #             SRRF_array[i][j] = pps

    #     return SRRF_array








            





