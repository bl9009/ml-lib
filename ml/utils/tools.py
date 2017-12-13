"""Utility functions for dealing with numpy arrays"""

import numpy as np

def instance_count(X):
    """Return number of instances of feature set."""
    # alternatively: return X.shape[0]
    return len(X)

def feature_count(X):
    """Return number of features of feature set."""
    return X.shape[1]

def class_count(y):
    """Return array with counts of each class in label set y."""
    return np.bincount(y)

def insert_intercept(X):
    """Prepend feature x0 = 1 to feature set X.
    Args:

    X: Feature set.

    Returns:
    Updated feature set with prepended feature x0 = 1
    """
    instances, features = X.shape

    features += 1

    new_X = np.ones((instances, features))

    new_X[0:, 1:] = X

    return new_X

def vectorize(func):
    """Decorator for vectorizing functions."""
    return np.vectorize(func)
