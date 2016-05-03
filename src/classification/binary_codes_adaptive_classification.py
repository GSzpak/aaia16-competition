from collections import defaultdict

import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA


def optimize_binary_codes(codes_matrix, labels):
    label_to_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[label].append(index)
    current_codes_matrix_T = codes_matrix.transpose()
    prev_codes_matrix_T = np.zeros(current_codes_matrix_T.shape)
    while current_codes_matrix_T != prev_codes_matrix_T:
        prev_codes_matrix_T = current_codes_matrix_T
        for code_bit_num, column in enumerate(prev_codes_matrix_T):
            num_equal_zero = sum([not bit for bit in column])
            num_equal_one = sum([bit for bit in column])
            for _, indices in label_to_indices.iteritems():
                num_equal_zero_in_class = sum([not column[i] for i in indices])
                num_equal_one_in_class = sum([column[i] for i in indices])
                num_equal_zero_outside_class = num_equal_zero - num_equal_zero_in_class
                num_equal_one_outside_class = num_equal_one - num_equal_one_in_class
                gradient_if_equal_zero = -(num_equal_one_in_class - num_equal_one_outside_class)
                gradient_if_equal_one = num_equal_zero_in_class - num_equal_zero_outside_class
                for i in indices:
                    if column[i]:
                        current_codes_matrix_T[code_bit_num, i] = int(1 - gradient_if_equal_one > 0)
                    else:
                        current_codes_matrix_T[code_bit_num, i] = int(-gradient_if_equal_zero > 0)
    return current_codes_matrix_T.transpose()


def get_discr_binary_codes(X, y, num_of_features):
    def binarize(X):
        return (1 + np.sign(X)) / 2
    pca = PCA(n_components=num_of_features)
    svm_classifier = svm.SVC(kernel='linear')
    original_dim = X.T.shape[0]
    B = pca.fit_transform(X)
    B = binarize(B)
    labels_matrix = 2 * B - 1
    hyperplanes_T = np.zeros(num_of_features, original_dim)
    while True:
        for i in xrange(num_of_features):
            current_labels = labels_matrix[:, i]
            svm_classifier.fit(X, current_labels)
            hyperplanes_T[i] = svm_classifier.coef_
        B_prim = binarize(hyperplanes_T * X.T)
        B_prim = optimize_binary_codes(B_prim, y)
        if B_prim == B:
            return hyperplanes_T


def discriminative_binary_codes_adaptive_classification(X_train, y_train, X_test, num_of_features):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    y_test = classifier.predict(X_test)
    while True:
        hyperplanes_tranposed = get_discr_binary_codes(X_test, y_test, num_of_features)
        new_X_train = np.sign(hyperplanes_tranposed * X_train)
        classifier.fit(new_X_train, y_train)
        hyperplane_train_orthogonal = classifier.coef_
        y_test_prim = np.sign(hyperplane_train_orthogonal * np.sign(hyperplanes_tranposed * X_train))
        if y_test_prim == y_test:
            break
        else:
            y_test = y_test_prim
    return y_test