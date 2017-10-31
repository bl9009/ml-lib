"""Implementation of Logistic Regression algorithm."""

import numpy as np

from sgdreg import SgdRegressor

class LogisticRegressor(SgdRegressor):
    """Logistic regression algorithm.

        Provides a logistic regression model using the sigmoid function.
    """

    def h(self, x):
        """Hypothesis function h(x) used to make predictions.

        Args:
            x: Vector of features to make prediction of.

        Returns:
            Vector of predictions.
        """
        z = self.theta.T.dot(x)

        return sigmoid(z)

def sigmoid(z):
    """Calculates sigmoid function."""
    return 1. / (1 + np.exp(-z))
