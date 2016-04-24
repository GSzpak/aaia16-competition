import pandas as pd
from pymongo import MongoClient
from sklearn import preprocessing

from src.import_data import DB_NAME
from helpers import ALL_FEATURES_KEY

# Standard scaling
STANDARD_SCALER_KEY = 'standard_scaled_features'
# Scaling to [0,1]
MIN_MAX_SCALER_KEY = 'zero_one_scaled_features'
# Scaling to [-1, 1]
MAX_ABS_SCALER_KEY = 'minus_one_one_scaled_features'

# Mapping: scaling function -> key in mongo
FUNCTION_TO_KEY = {
    preprocessing.scale: STANDARD_SCALER_KEY,
    preprocessing.minmax_scale: MIN_MAX_SCALER_KEY,
    preprocessing.maxabs_scale: MAX_ABS_SCALER_KEY,
}


def do_scale_features(collection, scaling_function):
    cursor = collection.find(filter={}, modifiers={"$snapshot": True})
    id_to_index = {}
    df_to_scale_rows = []
    for obj in cursor:
        id_to_index[obj['_id']] = len(df_to_scale_rows)
        df_to_scale_rows.append([ALL_FEATURES_KEY])
    cursor.close()
    df_to_scale = pd.DataFrame(df_to_scale_rows)
    scaled_df = scaling_function(df_to_scale)
    scaled_df.columns = df_to_scale.columns
    mongo_key = FUNCTION_TO_KEY[scaling_function]
    cursor = collection.find(filter={}, modifiers={"$snapshot": True})
    for obj in cursor:
        index_in_df = id_to_index[obj['_id']]
        scaled_features = {feature_name: scaled_df.ix[index_in_df, feature_name]
                           for feature_name in scaled_df.columns}
        obj[mongo_key] = scaled_features
        collection.save(obj)
    cursor.close()


def scale_features(scaling_function=preprocessing.scale):
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data, db.test_data]
        for collection in collections:
            do_scale_features(collection, scaling_function)
