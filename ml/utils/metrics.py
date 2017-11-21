"""Functions for error and loss calculation."""

import math

import numpy as np

from . import numpy_utils as np_utils

def mse(y, y_predicted):
    """Calculates MSE of given target and predicted values.

    Args:
        y: Ground truth.
        y_predicted: Predictions.

    Returns:
        The MSE.
    """
    m = y.size

    return (1./m) * sum(np.square(y_predicted - y))

def rmse(y, y_predicted):
    """Calculates RMSE of given target and predicted values.

    Args:
        y: Ground truth.
        y_predicted: Predictions.

    Returns:
        The RMSE.
    """
    return math.sqrt(mse(y, y_predicted))

def rss(y, y_predicted):
    """Calculates RSS (residual sum of squares) for given target
    and predicted values.

    Args:
        y: Ground truth.
        y_predicted: Predicted values.

    Returns:
        Residual sum of squares.
    """
    return sum((y_predicted - y) ** 2)

def log_loss(y, y_predicted):
    """Calculate the log loss function for given target and
    predicted values.

    This is used for estimating the loss for logistic models.

    Args:
        y: Ground truth.
        y_predicted: Predicted values.

    Returns:
        Log loss error.
    """
    m = y.size

    return (-1./m) * sum(y.T.dot(np.log(y_predicted))
                         + (1-y).T.dot(np.log(1-y_predicted)))

def gini(X, y):
    """Calculate gini impurity of given data set X.

    Args:
        X: Training data set.
        y: Label set

    Returns:
        Gini impurity as float.
    """
    m = np_utils.instance_count(X)

    label_counts = np_utils.label_counts(y)

    return 1 - sum([(n / m)**2 for n in label_counts])
