import numpy as np


def interpolate(x: np.array, magnification: int):  # adapted from:https://github.com/scipy/scipy/blob/v1.9.3/scipy/signal
    reshaped = x
    for axis in [0, 1]:
        num = x.shape[axis]*magnification
        nx = x.shape[axis]
        X = np.fft.fft(reshaped, axis=axis).real
        new_shape = list(reshaped.shape)
        new_shape[axis] = num // 2 + 1
        Y = np.zeros(new_shape, X.dtype)
        nyq = nx // 2 + 1  # Slice index that includes Nyquist if present
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(0, nyq)
        Y[tuple(sl)] = X[tuple(sl)]
        if nx % 2 == 0:
            if nx < num:
                sl[axis] = slice(nx//2, nx//2 + 1)
                Y[tuple(sl)] *= 0.5
        y = np.fft.ifft(Y, num, axis=axis).real
        reshaped = y
    y = (y - np.min(y))/(np.max(y)-np.min(y))
    return y