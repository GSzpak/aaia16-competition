import itertools

import pandas
from pymongo import MongoClient

from src.import_data import DB_NAME
from working_site_features import CATEGORICAL_FEATURES


FEATURES_TO_BINARIZE = [
    'latest_seismic_assessment',
    'latest_seismoacoustic_assessment',
    'latest_comprehensive_assessment',
    'latest_hazards_assessment',
    'geological_assessment'
] + CATEGORICAL_FEATURES
BINARIZED_FEATURES_KEY = 'binarized_features'


def save_binarized(collection, binarized_df, obj_id_to_index):
    print "Saving binarized features"
    counter = 0
    for _id, index_in_df in obj_id_to_index.iteritems():
        binarized_features = {feature_name: binarized_df.ix[index_in_df, feature_name]
                              for feature_name in binarized_df.columns}
        collection.update_one({
            '_id': _id
        }, {
            '$set': {
                BINARIZED_FEATURES_KEY: binarized_features
            }
        }, upsert=False)
        if counter % 1000 == 0:
            print "Progress: {}".format(counter)
        counter += 1


def do_binarize_features(collection):
    cursor = collection.find(filter={}, modifiers={"$snapshot": True})
    cursor.batch_size(10000)
    obj_id_to_index = {}
    rows_to_binarize = []
    print "Reading features to binarize"
    for counter, obj in itertools.izip(itertools.count(), cursor):
        obj_id_to_index[obj['_id']] = len(rows_to_binarize)
        current_row = {feature_name: obj[feature_name] for feature_name in FEATURES_TO_BINARIZE}
        rows_to_binarize.append(current_row)
        if counter % 1000 == 0:
            print "Progress: {}".format(counter)
    cursor.close()
    df = pandas.DataFrame(rows_to_binarize)
    binarized_df = pandas.get_dummies(df, prefix_sep='=')
    print list(binarized_df.columns)
    save_binarized(collection, binarized_df, obj_id_to_index)


def binarize_features():
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        for collection in collections:
            do_binarize_features(collection)
