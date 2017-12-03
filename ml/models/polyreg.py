"""Polynomial regression implementations."""

from itertools import combinations_with_replacement as combinations_w_r
from itertools import chain

from math import factorial as fac

import numpy as np

from ..utils import tools

from .linearreg import LinearRegressor, LinearSgdRegressor

class PolynomialRegressor(LinearRegressor):
    """Polynomaial Regression that is trained on polynomial features. A linear
    regression model based on the normal equation is used for training and
    prediction.

    Attributes:
        degree: Order of the polynomial features that will be crated.
        alpha: Regularization factor for ridge regularization.
    """
    def __init__(self, degree=1, alpha=0.):
        self.degree = degree

        super(PolynomialRegressor, self).__init__(alpha)

    def fit(self, X, y):
        """Train the model with polynomials generated out of the given features.

        Args:
            X: Features, which will be used to genrate polynomial features.
            y: Labels for training, the ground truth.
        """
        X_poly = polynomial_features(X, self.degree)

        return super(PolynomialRegressor, self).fit(X_poly, y)

    def predict(self, X):
        """Perform predictions based on fitted model.

        Args:
            X: Feature set to predict values for.

        Returns:
            Numpy array with the predicted results.
        """
        X_poly = polynomial_features(X, self.degree)

        return super(PolynomialRegressor, self).predict(X_poly)

class PolynomialSgdRegressor(LinearSgdRegressor):
    """Polynomaial Regression that is trained on polynomial features. A linear
    regression model based on stochastic gradient descent is used for training
    and prediction.

    Attributes:
        degree: Order of the polynomial features that will be crated.
        eta0: Starting learning rate.
        annealing: Rate for annealing the learning rate.
        epochs: Number of epochs used for training.
        alpha: Regularization factor
        l1_ratio: l1 penalty ratio
    """
    def __init__(self, degree=1, eta0=0.01, annealing=0.25, epochs=100, alpha=0., l1_ratio=1.):
        self.degree = degree

        super(PolynomialSgdRegressor, self).__init__(eta0, annealing, epochs, alpha, l1_ratio)

    def fit(self, X, y):
        """Train the model with polynomials generated out of the given features.

        Args:
            X: Features, which will be used to genrate polynomial features.
            y: Labels for training, the ground truth.
        """
        X_poly = polynomial_features(X, self.degree)

        super(PolynomialSgdRegressor, self).fit(X_poly, y)

    def predict(self, X):
        """Perform predictions based on fitted model.

        Args:
            X: Feature set to predict values for.

        Returns:
            Numpy array with the predicted results.
        """
        X_poly = polynomial_features(X, self.degree)

        return super(PolynomialSgdRegressor, self).predict(X_poly)

def polynomial_features(X, degree):
    """Generate polynomial features of the given degree.

    Args:
        X: Feature set to generate polynomial features of.
        degree: Degree of generated polynomial features.

    Returns:
        A numpy array containing the polynomial feature set.
    """
    m = tools.instance_count(X)
    n = tools.feature_count(X)

    n_poly, combinations = __generate_combinations(n, degree)

    X_poly = np.zeros((m, int(n_poly)))

    for i, combination in enumerate(combinations):
        tmp = 1

        for j in combination:
            tmp *= X[:, j:j+1]

        X_poly[:, i:i+1] = tmp

    return X_poly

def __generate_combinations(feature_count, degree):
    feature_indices = range(feature_count)

    combinations = [combinations_w_r(feature_indices, i) for i in range(1, degree+1)]

    # (n + d)! / n!d! - 1
    n = fac(feature_count + degree) / (fac(degree) * fac(feature_count)) - 1

    return n, chain.from_iterable(combinations)
