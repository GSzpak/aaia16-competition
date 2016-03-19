import numpy as np

from sklearn.neighbors import DistanceMetric


def multidimensional_dynamic_time_warping(md_time_series1, md_time_series2, distance_fun=abs):
    if md_time_series1.shape[0] <= 24 or md_time_series2.shape[0] <= 24:
        num_of_dimensions = 1
    else:
        num_of_dimensions = md_time_series1.shape[0] / 24
    len1 = md_time_series1.shape[0] / num_of_dimensions
    len2 = md_time_series2.shape[0] / num_of_dimensions
    distances = np.zeros((len1, len2))
    for i in xrange(len1):
        for j in xrange(len2):
            distances[i, j] = sum([distance_fun(md_time_series1[i + k * len1] - md_time_series2[j + k * len2])
                                   for k in xrange(num_of_dimensions)])
    accumulated_cost = np.zeros((len1, len2))
    accumulated_cost[0, 0] = distances[0, 0]
    for i in xrange(1, len1):
        accumulated_cost[i, 0] = accumulated_cost[i - 1, 0] + distances[i, 0]
    for j in xrange(1, len2):
        accumulated_cost[0, j] = accumulated_cost[0, j - 1] + distances[0, j]
    for i in xrange(1, len1):
        for j in xrange(len2):
            accumulated_cost[i, j] = min(accumulated_cost[i - 1, j - 1],
                                         accumulated_cost[i, j - 1],
                                         accumulated_cost[i - 1, j]) + distances[i, j]
    return accumulated_cost[len1 - 1, len2 - 1]


md_dtw = DistanceMetric.get_metric('pyfunc', func=multidimensional_dynamic_time_warping)


def polygon_plot_distance(time_series1, time_series2):
    raise NotImplementedError


def multidimensional_polygon_plot_distance(md_time_series1, md_time_series2):
    # number of dimensions must be equal
    assert md_time_series1.shape[0] == md_time_series2.shape[0]
    num_of_dimensions = md_time_series1.shape[0]
    distance = 0.0
    for i in xrange(num_of_dimensions):
        time_series1 = md_time_series1[i, :]
        time_series2 = md_time_series1[i, :]
        distance += polygon_plot_distance(time_series1, time_series2)
    return distance