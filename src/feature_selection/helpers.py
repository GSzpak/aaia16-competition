import copy

from pymongo import MongoClient

from src.import_data import DB_NAME
from extraction import (
    CROSS_CORRELATIONS_FEATURES_KEY,
    FEATURES_KEY,
    ADDITIONAL_FEATURES,
    FEATURE_VALUE_KEY
)
from binarization import BINARIZED_FEATURES_KEY


ALL_FEATURES_KEY = 'features'


def get_sequence_features(sequence_name, sequence_info):
    features = copy.copy(sequence_info[CROSS_CORRELATIONS_FEATURES_KEY])
    for feature_name, feature_info in sequence_info[FEATURES_KEY].iteritems():
        current_key = "{}_{}".format(sequence_name, feature_name)
        features[current_key] = feature_info[FEATURE_VALUE_KEY]
    return features


def do_reorganize_features(collection):
    cursor = collection.find(filter={}, modifiers={"$snapshot": True})
    for obj in cursor:
        all_features = copy.copy(obj[BINARIZED_FEATURES_KEY])
        for sequence_name, sequence_info in obj['sequences'].iteritems():
            all_features.update(get_sequence_features(sequence_name, sequence_info))
        all_features.update({feature_name: obj[feature_name] for feature_name in ADDITIONAL_FEATURES})
        obj[ALL_FEATURES_KEY] = all_features
        collection.save(obj)
    cursor.close()


def reorganize_features():
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        for collection in collections:
            do_reorganize_features(collection)