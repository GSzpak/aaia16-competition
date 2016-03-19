import heapq
import itertools

import numpy as np
import pickle

from pymongo import MongoClient
from sklearn.neighbors import NearestNeighbors

from src.import_data import DB_NAME


KNN_LIMIT = 25


def get_md_time_series(mongo_obj, series_names, values_key):
    if series_names is None:
        series = mongo_obj['sequences'].values()
    else:
        series = [mongo_obj['sequences'][series_name] for series_name in series_names]
    result = []
    for s in series:
        result.extend(s[values_key])
    return np.array(result, dtype=np.float64)


def custom_calculate_knn(collection, distance_fun, series_names=None, values_key='z_normalization_values'):
    series_names_key = "_".join(series_names) if series_names is not None else "all"
    knn_key = "knn_{}_{}_{}".format(distance_fun.__name__, series_names_key, values_key)
    cursor1 = collection.find(filter={}, modifiers={"$snapshot": True})
    for obj1 in cursor1:
        neighbours_distances = []
        cursor2 = collection.find(filter={}, modifiers={"$snapshot": True})
        for obj2 in cursor2:
            time_series1 = get_md_time_series(obj1, series_names, values_key)
            time_series2 = get_md_time_series(obj2, series_names, values_key)
            distance = distance_fun(time_series1, time_series2)
            neighbours_distances.append((distance, obj2["_id"]))
        nearest_neighbours = heapq.nsmallest(KNN_LIMIT, neighbours_distances)
        nearest_neighbours.sort()
        obj1[knn_key] = nearest_neighbours
        collection.save(obj1)
        cursor2.close()
    cursor1.close()


def calculate_knn(collection, metric, series_names=None, values_key='z_normalization_values'):
    series_names_key = "_".join(series_names) if series_names is not None else "all"
    knn_key = "knn_{}_{}_{}".format("dtw", series_names_key, values_key)
    cursor = collection.find(projection={'sequences': True}, modifiers={"$snapshot": True})
    cursor.batch_size = 1000000
    all_time_series = []
    counter = 0
    for obj in cursor:
        time_series = get_md_time_series(obj, series_names, values_key)
        all_time_series.append(time_series)
        counter += 1
        print counter
    cursor.close()
    all_time_series = np.array(all_time_series)
    print all_time_series
    print all_time_series.shape
    neighbours = NearestNeighbors(
        n_neighbors=KNN_LIMIT,
        algorithm='ball_tree',
        leaf_size=100,
        metric=metric,
        n_jobs=-1,
    )
    print "fitting..."
    neighbours.fit(all_time_series)
    with open("{}_neighours.p".format(knn_key), "wb") as f:
        pickle.dump(neighbours, f)
    print "finished"


def knn(metric, series_names=None, values_key='z_normalization_values'):
    with MongoClient() as client:
        db = client[DB_NAME]
        collections = [db.training_data]
        for collection in collections:
            calculate_knn(
                collection,
                metric,
                series_names=series_names,
                values_key=values_key,
            )