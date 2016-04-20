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
    return np.min(time_series)


@return_dict
def max(time_series):
    return np.max(time_series)


@return_dict
def quantiles(time_series):
    return {
        'quantile_25': np.percentile(time_series, 25, interpolation='midpoint'),
        'median': np.percentile(time_series, 50, interpolation='midpoint'),
        'quantile_75': np.percentile(time_series, 75, interpolation='midpoint'),
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


def autocorrelation(time_series, k):
    return cross_correlation(time_series, time_series, k)


@return_dict
def autocorrelations_8hr(time_series):
    return {
        'autocorrelation_8hr': autocorrelation(time_series, 8),
        'autocorrelation_16hr': autocorrelation(time_series, 16),
    }


def cross_correlation(time_series1, time_series2, k):
    def std_estimate(time_series, mean_, length):
        return np.sqrt(sum([(time_series[i] - mean_) ** 2 for i in xrange(length)]))
    assert len(time_series1) == len(time_series2)
    length = len(time_series1)
    mean1 = np.mean(time_series1)
    mean2 = np.mean(time_series2)
    numerator = sum([(time_series1[i] - mean1) * (time_series2[i - k] - mean2) for i in xrange(k, length)])
    denominator = std_estimate(time_series1, mean1, length) * std_estimate(time_series2, mean2, length)
    return numerator / denominator if denominator != 0.0 else 0.0


AGGREGATION_FUNCTIONS = [mean, std, min, max, quantiles, linear_weighted_average, quadratic_weighted_average,
                         arg_max, arg_min, avg_derivatives, avg_integrals, kurtosis, standard_mean_error,
                         mean_absolute_deviation, median_absolute_deviation, autocorrelations_8hr]

# TODO: DWT, DFT