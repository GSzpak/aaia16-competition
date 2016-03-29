import aggregation_functions

from pymongo import MongoClient

from src.import_data import DB_NAME


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
    time_series = time_series_info['values']
    extraction_funs = [function for _, function in aggregation_functions.__dict__.iteritems() if callable(function)]
    features = get_all_features(time_series, extraction_funs)
    features.update(calculate_8hr_aggregates(time_series, time_series_name))
    last_8hr_features = get_all_features(time_series[-8:], extraction_funs)
    last_8hr_features = {
        '{}_{}'.format('last_8hr', feature_name): feature_value
        for feature_name, feature_value in last_8hr_features.iteritems()
    }
    features.update(last_8hr_features)
    time_series_info['values_features'] = features


def add_features():
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        for collection in collections:
            cursor = collection.find(filter={}, modifiers={"$snapshot": True})
            for obj in cursor:
                all_time_series = obj['sequences']
                for time_series_name, time_series_info in all_time_series.iteritems():
                    do_add_features(time_series_name, time_series_info)
                collection.save(obj)
            cursor.close()