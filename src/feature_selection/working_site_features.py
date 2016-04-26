import csv
import itertools
import os

from pymongo import MongoClient

from src.import_data import DB_NAME, DATA_DIR


WORKING_SITE_FILE = 'working_site_metadata.csv'
ALL_FEATURES = [
    'main_working_id',
    'main_working_name',
    'region_name',
    'bed_name',
    'main_working_type',
    'main_working_height',
    'geological_assessment',
]
CONTINUOUS_FEATURES = ['main_working_height']
CATEGORICAL_FEATURES = ['geological_assessment']


def read_working_site_dict():
    result = {}
    file_path = os.path.join(DATA_DIR, WORKING_SITE_FILE)
    with open(file_path, "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            current_working_site_data = {feature_name: value for feature_name, value in zip(ALL_FEATURES, row)}
            working_id = current_working_site_data.pop(ALL_FEATURES[0])
            result[working_id] = current_working_site_data
    return result


def do_add_working_site_features(collection, working_site_data):
    cursor = collection.find(filter={}, modifiers={"$snapshot": True})
    for obj in cursor:
        working_site_id = obj['main_working_id']
        current_working_data = working_site_data[working_site_id]
        for feature_name in itertools.chain(CONTINUOUS_FEATURES, CATEGORICAL_FEATURES):
            obj[feature_name] = current_working_data[feature_name]
        collection.save(obj)
    cursor.close()


def add_working_site_features():
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        working_site_data = read_working_site_dict()
        for collection in collections:
            do_add_working_site_features(collection, working_site_data)
