"""Implementation of Linear Regression models."""

import numpy as np

from ..utils import tools
from .sgdreg import SgdRegressor

class LinearRegressor(object):
    """Implements a Linear Regression model based on normal equation learning.

    When specifying regularization factor alpha, model performs ridge
    regression.

    Attributes:
        theta: Paramaters for linear hypothesis.
    """
    def __init__(self, alpha=0.):
        """Initializes Regressor.

        Args:
            alpha: Regularization factor for ridge regularization.
        """
        self.alpha = alpha

        self.theta = None

    def fit(self, X, y):
        """Fits the model.

        Args:
            X: Training data set.
            y: Labels.
        """
        X = tools.insert_intercept(X)

        A = np.identity(tools.feature_count(X))

        self.theta = np.linalg.inv(X.T.dot(X) + self.alpha * A).dot(X.T.dot(y)).T

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        X = tools.insert_intercept(X)

        return self.h(X)

    def h(self, x):
        """Hypothesis function h(x) used to make predictions.

        Args:
            x: Vector of features to make prediction of.

        Returns:
            Vector of predictions.
        """
        return np.asmatrix(x).dot(self.theta.T)


class LinearSgdRegressor(SgdRegressor):
    """Linear Regression model using stochastic gradient descent for learning."""

    def h(self, x):
        """Hypothesis function h(x) used to make predictions.

        Args:
            x: Vector of features to make prediction of.

        Returns:
            Vector of predictions.
        """
        return np.asmatrix(x).dot(self.theta.T)
