from collections import defaultdict
from itertools import izip, count

from pymongo import MongoClient
import numpy as np
import pandas as pd

import aggregation_functions
from src.import_data import DB_NAME
from src.parse_data import SEQ_COLUMN_NAMES

#TODO:
# - DFT, DWT,
# auto-correlations,
# cross-correlations


EIGHT_HOUR_AGGREGATE_FUNS = {
    'count_e2': sum,
    'count_e3': sum,
    'count_e4': sum,
    'count_e5': sum,
    'count_e6plus': sum,
    'sum_e2': sum,
    'sum_e3': sum,
    'sum_e4': sum,
    'sum_e5': sum,
    'sum_e6plus': sum,
    'total_number_of_bumps': sum,
    'number_of_rock_bursts': sum,
    'number_of_destressing_blasts': sum,
    'highest_bump_energy': max,
    'max_gactivity': max,
    'max_genergy': max,
    'avg_gactivity': aggregation_functions.mean,
    'avg_genergy': aggregation_functions.mean,
    'max_difference_in_gactivity': max,
    'max_difference_in_genergy': max,
    'avg_difference_in_gactivity': aggregation_functions.mean,
    'avg_difference_in_genergy': aggregation_functions.mean,
}

def calculate_8hr_aggregates(time_series, time_series_name):
    assert len(time_series) == 24
    chunk_size = 8
    chunk1, chunk2, chunk3 = [time_series[i:i+chunk_size] for i in range(0, len(time_series), chunk_size)]
    aggregate_fun = EIGHT_HOUR_AGGREGATE_FUNS[time_series_name]
    return {
        '8hr_aggregates_1': aggregate_fun(chunk1),
        '8hr_aggregates_2': aggregate_fun(chunk2),
        '8hr_aggregates_3': aggregate_fun(chunk3),
    }


def get_all_features(time_series, extraction_funs):
    features = {}
    for extraction_fun in extraction_funs:
        features.update(extraction_fun(time_series))
    return features


def do_add_features(time_series_name, time_series_info):
    time_series = map(float, time_series_info['values'])
    features = get_all_features(time_series, aggregation_functions.AGGREGATION_FUNCTIONS)
    features.update(calculate_8hr_aggregates(time_series, time_series_name))
    last_8hr_features = get_all_features(time_series[-8:], aggregation_functions.AGGREGATION_FUNCTIONS)
    last_8hr_features = {
        '{}_{}'.format('last_8hr', feature_name): feature_value
        for feature_name, feature_value in last_8hr_features.iteritems()
    }
    features.update(last_8hr_features)
    features = {name: {'value': value} for name, value in features.iteritems()}
    time_series_info['values_features'] = features


def cross_correlation_8hr(time_series_name1, time_series_name2, time_series_info1, time_series_info2):
    time_series1 = map(float, time_series_info1['values'])
    time_series2 = map(float, time_series_info2['values'])
    return {
        '{}_{}_cross_correlation_8hr'.format(time_series_name1, time_series_name2):
            aggregation_functions.cross_correlation(time_series1, time_series2, 8),
        '{}_{}_cross_correlation_16hr'.format(time_series_name1, time_series_name2):
            aggregation_functions.cross_correlation(time_series1, time_series2, 16)
    }


def add_cross_correlations(all_time_series):
    for time_series_name1, time_series_info1 in all_time_series.iteritems():
        cross_correlations = {}
        for time_series_name2, time_series_info2 in all_time_series.iteritems():
            if time_series_name1 == time_series_name2:
                # Autocorrelation is kept separately
                continue
            cross_correlations.update(cross_correlation_8hr(
                time_series_name1,
                time_series_name2,
                time_series_info1,
                time_series_info2
            ))
        time_series_info1['cross_correlations'] = cross_correlations


def add_features():
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        for collection in collections:
            cursor = collection.find(filter={}, modifiers={"$snapshot": True})
            counter = 0
            for obj in cursor:
                all_time_series = obj['sequences']
                for time_series_name, time_series_info in all_time_series.iteritems():
                    do_add_features(time_series_name, time_series_info)
                add_cross_correlations(all_time_series)
                collection.save(obj)
                counter += 1
                print counter
                if counter % 1000 == 0:
                    print "Progress: {}".format(counter)
            cursor.close()


def do_scale_features(collection, sequence_name):
    def minus_one_to_one_scaling(min_, max_, val):
        numerator = val - min_
        denominator = max_ - min_
        return 2 * (numerator / denominator) - 1
    projection_key = "sequences.{}".format(sequence_name)
    cursor = collection.find(projection={projection_key: True}, modifiers={"$snapshot": True})
    features = defaultdict(list)
    for obj in cursor:
        for key, val in obj['sequences'][sequence_name]['values_features'].iteritems():
            features[key].append(float(val['value']))
    cursor.close()
    for feature_name, feature_values in features.iteritems():
        max_ = np.max(feature_values)
        min_ = np.min(feature_values)
        if max_ <= 1 and min_ >= -1:
            continue
        for i in xrange(len(feature_values)):
            scaled_value = minus_one_to_one_scaling(min_, max_, feature_values[i])
            assert abs(scaled_value) <= 1
            feature_values[i] = scaled_value
    cursor = collection.find(projection={projection_key: True}, modifiers={"$snapshot": True})
    features = defaultdict(list)
    for i, obj in izip(count(), cursor):
        features_dict = obj['sequences'][sequence_name]['values_features']
        for feature_name, feature_values in features.iteritems():
            features_dict['scaled_value'] = feature_values[i]
        collection.save(obj)
    cursor.close()


def scale_features():
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        for collection in collections:
            cursor = collection.find(filter={}, modifiers={"$snapshot": True})
            counter = 0
            for obj in cursor:
                all_time_series = obj['sequences']
                for time_series_name, time_series_info in all_time_series.iteritems():
                    do_add_features(time_series_name, time_series_info)
                add_cross_correlations(all_time_series)
                collection.save(obj)
                counter += 1
                print counter
                if counter % 1000 == 0:
                    print "Progress: {}".format(counter)
            cursor.close()