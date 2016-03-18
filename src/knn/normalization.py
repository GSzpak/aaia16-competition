import numpy as np

from pymongo import MongoClient

from src.import_data import DB_NAME


def standard_normalization(time_series):
    time_series = np.asarray(time_series, dtype=np.float64)
    min_ = np.min(time_series)
    max_ = np.max(time_series)
    return [(val - min_) / (max_ - min_) if max_ - min_ > 0 else 0
            for val in np.nditer(time_series)]


def z_normalization(time_series):
    time_series = np.asarray(time_series, dtype=np.float64)
    mean = np.mean(time_series)
    stdev = np.std(time_series)
    if stdev > 0:
        return [(val - mean) / stdev for val in np.nditer(time_series)]
    else:
        return time_series.tolist()


def do_normalize(collection, normalization_fun):
    cursor = collection.find(filter={}, modifiers={"$snapshot": True})
    key = '{}_values'.format(normalization_fun.__name__)
    for obj in cursor:
        sequences = obj['sequences']
        for sequence_name, sequence in sequences.iteritems():
            normalized_time_series = normalization_fun(sequence['values'])
            sequence[key] = normalized_time_series
        collection.save(obj)
    cursor.close()


def normalize():
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        normalization_functions = [standard_normalization, z_normalization]
        for collection in collections:
            for normalization_fun in normalization_functions:
                do_normalize(collection, normalization_fun)