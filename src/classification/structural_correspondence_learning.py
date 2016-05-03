import numpy as np


def get_most_often_occuring_features(matrix, num_of_columns):
    """
    :param X: binary matrix
    :return: List of (index, number of nonzero values) pairs for
                columns with largest number of non-zero values
    """
    matrix_transposed = matrix.transpose()
    columns_with_indices = enumerate(matrix_transposed)
    sorted_columns_with_indices = sorted(
        columns_with_indices,
        key=lambda (index, column): sum(column),
        reverse=True
    )
    result = sorted_columns_with_indices[:num_of_columns]
    return [(index, sum(column)) for (index, column) in result]


def most_often_occuring_pivot_features(X_train, X_test, num_of_features):
    """
    Assumes, that all features are binary
    """
    num_of_training_examples = float(X_train.shape[0])
    num_of_testing_examples = float(X_test.shape[0])
    scaling_parameter = num_of_training_examples / num_of_testing_examples
    training_pivots = get_most_often_occuring_features(X_train, num_of_features)
    test_pivots = get_most_often_occuring_features(X_test, num_of_features)
    test_pivots = [(index, scaling_parameter * score) for (index, score) in test_pivots]
    all_pivots = training_pivots + test_pivots
    all_pivots.sort(key=lambda (index, score): score, reverse=True)
    result = all_pivots[:num_of_features]
    return [index for (index, score) in result]


def structural_correspondence_transform(X_train, X_test, loss_function,
                                        choose_pivot_features_fun, num_of_pivot_features,
                                        lambda_2_reg=0.0001):
    pivot_features_indices = choose_pivot_features_fun(X_train, X_test, num_of_pivot_features)
    labels_for_pivot = {}
