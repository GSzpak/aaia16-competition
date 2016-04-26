from os import path

from pymongo import MongoClient

from parse_data import read_data


DB_NAME = 'aaia_competition'
TRAINING_COLLECTION = 'training_data'
TEST_COLLECTION = 'test_data'
BASE_DIR = path.dirname(path.dirname(__file__))
DATA_DIR = path.join(BASE_DIR, 'data')
TRAINING_DATA = [('trainingData.csv', 'trainingLabels.csv')] + \
                [('additional_training_data_{}.csv'.format(i), "additional_training_labels_{}.csv".format(i))
                    for i in xrange(1, 5)]
TEST_DATA = ["testData.csv"]


def clear_db():
    with MongoClient() as client:
        db = client[DB_NAME]
        db['training_data'].drop()
        db['test_data'].drop()


def import_data():
    clear_db()
    with MongoClient() as client:
        db = client[DB_NAME]
        training_collection = db['training_data']
        test_collection = db['test_data']
        for data_file, labels_file in TRAINING_DATA:
            read_data(
                data_file_path=path.join(DATA_DIR, data_file),
                db_collection=training_collection,
                labels_file_path=path.join(DATA_DIR, labels_file),
            )
        for data_file in TEST_DATA:
            read_data(
                data_file_path=path.join(DATA_DIR, data_file),
                db_collection=test_collection
            )