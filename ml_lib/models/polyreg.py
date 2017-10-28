"""Polynomial regression implementations."""

import numpy as np

from itertools import combinations_with_replacement as combinations_w_r
from itertools import chain

from math import factorial as fac

from ..utils import numpy_utils as np_utils

from .linearreg import LinearRegressor, SgdRegressor

class PolynomialLinearRegressor(LinearRegressor):
    """Polynomaial Regression that is trained on polynomial features. A linear
    regression model based on the normal equation is used.

    Attributes:
        degree: Order of the polynomial features that will be crated.
        alpha: Regularization factor for ridge regularization.
    """
    def __init__(self, degree=1, alpha=0.):
        self.degree = degree

        super(LinearRegressor, self).__init__(alpha)

    def fit(self, X, y):
        """Train the model with polynomials generated out of the given features.

        Args:
            X: Features, which will be used to genrate polynomial features.
            y: Labels for training, the ground truth.
        """
        X_poly = polynomial_features(X, self.degree)

        return super(LinearRegressor, self).fit(X_poly, y)

    def predict(self, X):
        """Perform predictions based on fitted model.

        Args:
            X: Feature set to predict values for.

        Returns:
            Numpy array with the predicted results.
        """
        X_poly = polynomial_features(X, self.degree)

        return super(LinearRegressor, self).predict(X)

#class PolynomialSgdRegressor(SgdRegressor):
#    pass

def polynomial_features(X, degree):
    m = np_utils.instance_count(X)
    n = np_utils.feature_count(X)

    n_poly, combinations = generate_combinations(n, degree)
    
    X_poly = np.zeros((m, int(n_poly)))

    for i, combination in enumerate(combinations):
        tmp = 1

        for j in combination:
            tmp *= X[:,j:j+1]

        X_poly[:,i:i+1] = tmp

    return X_poly

def generate_combinations(feature_count, degree):
    feature_indices = range(feature_count)

    combinations = [combinations_w_r(feature_indices, i) for i in range(1, degree+1)]
    
    # (n + d)! / n!d! - 1
    n = fac(feature_count + degree) / (fac(degree) * fac(feature_count)) - 1

    return n, chain.from_iterable(combinations)
