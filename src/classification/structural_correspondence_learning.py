import numpy as np

from sklearn.linear_model import SGDRegressor


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


def get_pivot_predictor(X, y, pivot_index, loss_function, lambda_2):
    # delete column with current pivot feature
    X = np.delete(X, pivot_index, axis=1)
    regressor = SGDRegressor(loss=loss_function, penalty='l2', alpha=lambda_2)
    regressor.fit(X, y)
    # FIXME: coefficient for removed feature?
    return regressor.coef_


def augment_matrix(X, pivot_predictors):
    assert pivot_predictors.shape[1] == X.shape[1]
    features_to_add = pivot_predictors * X.T
    np.append(X, features_to_add, axis=1)


def structural_correspondence_transform(X_train, X_test, choose_pivot_features_fun, num_of_pivot_features,
                                        loss_function='huber', lambda_2_reg=0.0001, approximate=False):
    pivot_features_indices = choose_pivot_features_fun(X_train, X_test, num_of_pivot_features)
    X = np.concatenate((X_train, X_test))
    num_of_examples = X.shape[0]
    labels_for_pivot = {
        pivot: [X[j, pivot] for j in xrange(num_of_examples)] for pivot in pivot_features_indices
    }
    pivot_predictors = []
    for pivot, labels in labels_for_pivot.iteritems():
        pivot_predictors.append(get_pivot_predictor(X, labels, pivot, loss_function, lambda_2_reg))
    pivot_predictors = np.asarray(pivot_predictors)
    if approximate:
        # Add SVD to reduce dimensionality
        pass
    augment_matrix(X_train, pivot_predictors)
    augment_matrix(X_test, pivot_predictors)
    return X_train, X_test
