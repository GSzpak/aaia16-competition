import numpy as np
from sklearn import svm


def optimize_binary_codes(B):
    pass


def get_discr_binary_codes(X, y, num_of_features):
    pass


def discriminative_binary_codes_adaptive_classification(X_train, y_train, X_test, num_of_features):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    y_test = classifier.predict(X_test)
    while True:
        A = get_discr_binary_codes(X_test, y_test, num_of_features)
        A_T = A.transpose()
        new_X_train = np.sign(A_T * X_train)
        classifier.fit(new_X_train, y_train)
        hyperplane_orthogonal = classifier.coef_
        y_test_prim = np.sign(hyperplane_orthogonal * np.sign(A_T * X_train))
        if y_test_prim == y_test:
            break
        else:
            y_test = y_test_prim
    return y_test