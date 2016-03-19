import functools
import heapq
import itertools
import multiprocessing
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


def find_nn(distance_fun, all_time_series, ts_tuple):
    (index, (id_, time_series)) = ts_tuple
    neighbours_distances = []
    for index2, (id2, time_series2) in enumerate(all_time_series):
        if index == index2:
            continue
        distance = distance_fun(time_series, time_series2)
        neighbours_distances.append((distance, id2))
    nearest_neighbours = heapq.nsmallest(KNN_LIMIT, neighbours_distances)
    nearest_neighbours.sort()
    print index
    return id_, nearest_neighbours


def custom_calculate_knn(collection, distance_fun, series_names=None, values_key='z_normalization_values'):
    series_names_key = "_".join(series_names) if series_names is not None else "all"
    knn_key = "knn_{}_{}_{}".format(distance_fun.__name__, series_names_key, values_key)
    cursor = collection.find(projection={'sequences': True}, modifiers={"$snapshot": True})
    all_time_series = []
    for obj in cursor:
        time_series = get_md_time_series(obj, series_names, values_key)
        all_time_series.append((obj["_id"], time_series))
    cursor.close()
    print "data loaded"
    pool = multiprocessing.Pool(processes=4)
    calc_func = functools.partial(find_nn, distance_fun, all_time_series)
    results = pool.map(calc_func, enumerate(all_time_series))
    pool.close()
    pool.join()
    with open("{}_results.p".format(knn_key), "wb") as f:
        pickle.dump(results, f)
    cursor = collection.find(filter={}, modifiers={"$snapshot": True})
    for obj, (id_, nearest_neighbours) in itertools.izip(cursor, results):
        assert obj["_id"] == id_
        obj[knn_key] = nearest_neighbours
        collection.save(obj)
    cursor.close()


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
            custom_calculate_knn(
                collection,
                metric,
                series_names=series_names,
                values_key=values_key,
            )
