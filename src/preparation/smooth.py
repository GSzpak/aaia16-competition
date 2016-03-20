import numpy


# wtf is going on here
def smooth(time_series, window_len=3, window='hanning'):
    if time_series.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if time_series.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return time_series
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = numpy.r_[2 * time_series[0] - time_series[window_len - 1::-1], time_series,
                 2 * time_series[-1] - time_series[-1:-window_len:-1]]
    if window == 'flat':
        # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.{}(window_len)'.format(window))
    y = numpy.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]
