from functools import wraps
import numpy as np


def return_dict(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, dict):
            return result
        else:
            return {
                function.__name__: result
            }
    return wrapper


def cross_correlation(time_series1, time_series2, k):
    def std_estimate(time_series, mean_, length):
        return np.sqrt(sum([(time_series[i] - mean_) ** 2 for i in xrange(length)]))
    assert len(time_series1) == len(time_series2)
    length = len(time_series1)
    mean1 = np.mean(time_series1)
    mean2 = np.mean(time_series2)
    numerator = sum([(time_series1[i] - mean1) * (time_series2[i - k] - mean2) for i in xrange(k, length)])
    denominator = std_estimate(time_series1, mean1, length) * std_estimate(time_series2, mean2, length)
    return numerator / float(denominator)


def autocorrelation(time_series, k):
    return cross_correlation(time_series, time_series, k)