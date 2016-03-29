from functools import wraps
import numpy as np
from scipy import stats

from utils import return_dict


@return_dict
def mean(time_series):
    return np.mean(time_series)


@return_dict
def std(time_series):
    return np.std(time_series)


@return_dict
def min(time_series):
    return min(time_series)


@return_dict
def max(time_series):
    return max(time_series)


@return_dict
def quantiles(time_series):
    return {
        'quantile_25': np.percentile(time_series, 25),
        'median': np.percentile(time_series, 50),
        'quantile_75': np.percentile(time_series, 75),
    }


@return_dict
def linear_weighted_average(time_series):
    return {
        'linear_avg': np.average(time_series, weights=[i + 1 for i in xrange(len(time_series))])
    }


@return_dict
def quadratic_weighted_average(time_series):
    return {
        'quadratic_avg': np.average(time_series, weights=[(i + 1) ** 2 for i in xrange(len(time_series))])
    }


@return_dict
def arg_max(time_series):
    return np.argmax(time_series) / float(len(time_series))


@return_dict
def arg_min(time_series):
    return np.argmin(time_series) / float(len(time_series))


@return_dict
def avg_derivatives(time_series):
    def derivative(sequence):
        return [sequence[i] - sequence[i - 1] for i in xrange(1, len(sequence))]
    first_derivative = derivative(time_series)
    second_derivative = derivative(first_derivative)
    return {
        'first_derivative_avg': np.mean(first_derivative),
        'second_derivative_avg': np.mean(second_derivative)
    }


@return_dict
def avg_integrals(time_series):
    def integral(sequence):
        return [(sequence[i] + sequence[i - 1]) / 2 for i in xrange(1, len(sequence))]
    first_integral = integral(time_series)
    second_integral = integral(first_integral)
    return {
        'first_integral_avg': sum(first_integral),
        'second_integral_avg': sum(second_integral)
    }


@return_dict
def kurtosis(time_series):
    return stats.kurtosis(time_series)


@return_dict
def standard_mean_error(time_series):
    return stats.sem(time_series)


@return_dict
def mean_absolute_deviation(time_series):
    return np.mean(np.absolute(time_series - np.mean(time_series)))


@return_dict
def median_absolute_deviation(time_series):
    return np.percentile(np.absolute(time_series - np.percentile(time_series, 50)), 50)


@return_dict
def autocorrelations_8hr(time_series):
    def autocorrelation(time_series, k):
        mean_ = mean(time_series)
        numerator = sum([(time_series[i] - mean_) * (time_series[i - k] - mean_) for i in xrange(k, len(time_series))])
        denominator = sum([(time_series[i] - mean_) * (time_series[i - k] - mean_) for i in xrange(k, len(time_series))])
        return numerator / float(denominator)
    return {
        'autocorrelation_8hr': autocorrelation(time_series, 8),
        'autocorrelation_16hr': autocorrelation(time_series, 16),
    }


# TODO: DWT, DFT