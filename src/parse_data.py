from itertools import izip
import numpy
import os


__all__ = ["read_data"]


META_COLUMN_NAMES = [
    'main_working_id',
    'total_bumps_energy',
    'total_tremors_energy',
    'total_destressing_blasts_energy',
    'total_seismic_energy',
    'latest_progress_estimation_l',
    'latest_progress_estimation_r',
    'latest_seismic_assessment',
    'latest_seismoacoustic_assessment',
    'latest_comprehensive_assessment',
    'latest_hazards_assessment',
    'latest_maximum_yield',
    'latest_maximum_meter',
]
SEQ_COLUMN_NAMES= [
    'count_e2',
    'count_e3',
    'count_e4',
    'count_e5',
    'count_e6plus',
    'sum_e2',
    'sum_e3',
    'sum_e4',
    'sum_e5',
    'sum_e6plus',
    'total_number_of_bumps',
    'number_of_destressing_blasts',
    'highest_bump_energy',
    'max_gactivity',
    'max_genergy',
    'avg_gactivity',
    'avg_genergy',
    'max_difference_in_gactivity',
    'max_difference_in_genergy',
    'avg_difference_in_gactivity',
    'avg_difference_in_genergy',
]
SEQUENCES_KEY = 'sequences'
SEQUENCE_LEN = 24
STAT_FUNCTIONS_NAMES = ["mean", "median", "min", "max"]


def read_metadata(input_row):
    result = {}
    for column_name, column_val in izip(META_COLUMN_NAMES, input_row):
        result[column_name] = column_val
    return result


def read_sequences(input_row):
    result = {}
    current_index = 0
    for column_name in SEQ_COLUMN_NAMES:
        result[column_name] = {}
        values = input_row[current_index:(current_index + SEQUENCE_LEN)]
        result[column_name]['values'] = values
        values = numpy.asarray(values, dtype=numpy.float64)
        for stat_fun_name in STAT_FUNCTIONS_NAMES:
            stat_fun = getattr(numpy, stat_fun_name)
            result[column_name][stat_fun_name] = stat_fun(values)
        assert len(result[column_name]['values']) == SEQUENCE_LEN
        current_index += SEQUENCE_LEN
    return result


def parse_row(data_row):
    data_row = data_row.splitlines()[0]
    data_row = data_row.split(',')
    result = {}
    meta_values = data_row[:len(META_COLUMN_NAMES)]
    sequence_values = data_row[len(META_COLUMN_NAMES):]
    result.update(read_metadata(meta_values))
    result[SEQUENCES_KEY] = read_sequences(sequence_values)
    return result


def read_training_data(data_file_path, db_collection, labels_file_path):
    file_name = os.path.basename(data_file_path)
    with open(data_file_path, "r") as data_file, open(labels_file_path, "r") as labels:
        for data_row, label in izip(data_file, labels):
            label = label.splitlines()[0]
            parsed_row = parse_row(data_row)
            parsed_row['file'] = file_name
            parsed_row['label'] = label
            db_collection.insert_one(parsed_row)


def read_test_data(data_file_path, db_collection):
    file_name = os.path.basename(data_file_path)
    with open(data_file_path, "r") as data_file:
        for data_row in data_file:
            parsed_row = parse_row(data_row)
            parsed_row['file'] = file_name
            db_collection.insert_one(parsed_row)


def read_data(data_file_path, db_collection, labels_file_path=None):
    if labels_file_path is None:
        read_test_data(data_file_path, db_collection)
    else:
        read_training_data(data_file_path, db_collection, labels_file_path)