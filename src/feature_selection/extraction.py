import itertools
import sys

import numpy as np
from pymongo import MongoClient

import aggregation_functions
from src.import_data import DB_NAME
from working_site_features import CONTINUOUS_FEATURES


CROSS_CORRELATIONS_FEATURES_KEY = 'cross_correlations'
FEATURES_KEY = 'values_features'
ADDITIONAL_FEATURES = [
    'main_working_id',
    'total_bumps_energy',
    'total_tremors_energy',
    'total_destressing_blasts_energy',
    'total_seismic_energy',
    'latest_progress_estimation_l',
    'latest_progress_estimation_r',
    'latest_maximum_yield',
    'latest_maximum_meter',
] + CONTINUOUS_FEATURES
FEATURE_VALUE_KEY = 'value'


#TODO:
# - DFT, DWT,
# auto-correlations - done, only for 8hr
# cross-correlations - done, only for 8hr


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
    'avg_gactivity': np.mean,
    'avg_genergy': np.mean,
    'max_difference_in_gactivity': max,
    'max_difference_in_genergy': max,
    'avg_difference_in_gactivity': np.mean,
    'avg_difference_in_genergy': np.mean,
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
    features = {name: {FEATURE_VALUE_KEY: value} for name, value in features.iteritems()}
    time_series_info[FEATURES_KEY] = features


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
        time_series_info1[CROSS_CORRELATIONS_FEATURES_KEY] = cross_correlations


def add_features():
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        for collection in collections:
            cursor = collection.find(filter={}, modifiers={"$snapshot": True})
            for counter, obj in itertools.izip(itertools.count(), cursor):
                all_time_series = obj['sequences']
                for time_series_name, time_series_info in all_time_series.iteritems():
                    do_add_features(time_series_name, time_series_info)
                add_cross_correlations(all_time_series)
                collection.save(obj)
                counter += 1
                if counter % 100 == 0:
                    print "Progress: {}".format(counter)
                    sys.stdout.flush()
            cursor.close()
