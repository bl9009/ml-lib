"""Implementation of Linear Regression models."""

import numpy as np

from ..utils import numpy_utils as np_utils

class LinearModel(object):
    """ Represents a linear model with a hypothesis of the form:
    theta0 + theta_1 * x_1 + .. + theta_j * x_j for j: 1..n

    Attributes:
        theta: Parameters of the linear hypothesis.
    """

    def __init__(self, dim=2):
        self.theta = np.ones((1, dim))

    def h(self, X):
        """ Calculates hypothesis for given feature matrix X.

        Args:
            X: A matrix of shape mxn where m is the number of instances and n
            is the number of features.

        Returns:
            The result of the hypothesis as float.
        """
        return np.asmatrix(X).dot(self.theta.T)

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
            alpha: Regularization factor for ridge regression
        """
        self.alpha = alpha

        self.model = LinearModel()

    def fit(self, X, y):
        """Fits the model using the normal equation.

        Args:
            X: A mxn matrix representing the training data set, where m is the
            number of training instances and n is the number of features.
            y: A vector with m values representing the training labels.
        """
        A = np.identity(np_utils.feature_count(X))

        self.model.theta = np.linalg.inv(X.T.dot(X) + self.alpha * A).dot(X.T.dot(y))

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: A matrix of shape mxn where m is the number of instances and n
            is the number of features.

        Returns:
            A numpy array containing the predicted values.
        """
        return self.model.h(X)


class SgdRegressor(object):
    """Linear Regression model using stochastic gradient descent for learning.

    Attributes:
        theta: Parameters for linear hypothesis.
    """

    def __init__(self, eta0=0.01, annealing=0., epochs=100, alpha=0., l1_ratio=1.):
        """Initializes Regressor with hyperparameters.

        Regularizes model with elastic net by setting regularization factor
        alpha. Setting l1_ratio to 1.0 performs l1 regularization (LASSO),
        l1_ratio to 0.0 for l2 regularization (ridge)

        Args:
            eta0: Starting learning rate.
            annealing: Rate for annealing the learning rate.
            epochs: Number of epochs used for training.
            alpha: Regularization factor
            l1_ratio: l1 penalty ratio
        """
        self.eta0 = eta0
        self.annealing = annealing
        self.epochs = epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        self.model = None

    def fit(self, X, y):
        """Fits the model using stochastic gradient descent algorithm.

        Args:
            X: Training data set.
            y: Labels.
        """
        X = np_utils.insert_intercept(X)

        self.model = LinearModel(np_utils.feature_count(X))

        m = np_utils.instance_count(X)

        gradient_vector = make_mse_gradient_vector(X, y)

        for epoch in range(self.epochs):
            for i in range(m):
                eta = self.__learning_schedule(epoch * m + i + 1)

                penalty = penalty_vector(self.model.theta, self.l1_ratio)
                gradient = gradient_vector(self.model.theta)

                self.model.theta = self.model.theta - eta * (gradient + self.alpha * penalty)

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        X = np_utils.insert_intercept(X)

        return self.model.h(X)

    def __learning_schedule(self, t):
        """Adjust learning rate depending on performed iterations t.

        Args:
            t: Number of iterations performed.

        Returns:
            Adjusted learning rate eta.
        """
        return self.eta0 / t**self.annealing


def make_h(theta):
    """Closure that returns hypothesis function h(x) with paramaters theta.

    Args:
        theta: Parameters for h(x).

    Returns:
        Hypothesis function h(x).
    """
    def h(x):
        """Hypothesis function h(x) used to make predictions.

        Args:
            x: Vector of features to make prediction of.

        Returns:
            Vector of predictions.
        """
        return np.asmatrix(x).dot(theta.T)

    return h

def make_mse_gradient_vector(X, y):
    """Closure that returns a function to calculate the MSE gradient vector.

    Args:
        X: Feature set.
        y: Labels.

    Returns:
        Function to calculate gradient vector of MSE function.
    """
    def mse_gradient_vector(theta):
        """Calculates the gradient vector of MSE function.

        Args:
            theta: Parameters to calculate MSE gradient for.

        Returns:
            Vector of gradients as numpy array.
        """
        m = np_utils.instance_count(X)

        h = make_h(theta)

        index = np.random.randint(m)

        x_i = X[index:index+1]
        y_i = y[index:index+1]

        return 2./m * x_i.T.dot(h(x_i) - y_i).T

    return mse_gradient_vector

def penalty_vector(theta, l1_ratio=1.):
    """Calculates vector of penalties for each paramater theta
       using elastic net method.

       l1_ratio specifies the ratio between l1 and l2 regression. Where 1.0 is
       equal to lasso regression, 0.0 to ridge regression and 0.5 for balanced
       lasso and ridge regression.

    Args:
        theta: Vector of parameters to penalize.
        l1_ratio: Ratio for l1 regularization.

    Returns:
        Numpy array with penalties for each parameter theta.
    """
    l1_penalty = lasso_vector(theta)
    l2_penalty = ridge_vector(theta)

    penalty = l1_ratio * l1_penalty + (1. - l1_ratio) / 2. * l2_penalty

    return penalty

def lasso_vector(theta):
    """Calculates gradient vector for LASSO regularization.

    Args:
        theta: Parameters to calculate gradient vector for.

    Returns:
        Numpy array with gradients.
    """
    return sign(theta)

@np_utils.vectorize
def sign(theta):
    """Calculates subgradient derivative for LASSO penalty."""
    if theta > 0:
        return 1
    if theta == 0:
        return 0
    if theta < 0:
        return -1

def ridge_vector(theta):
    """Calculates gradient vector for ridge regularization.

    Args:
        theta: Parameters to calculate gradient vector for.

    Returns:
        Numpy array with gradients.
    """
    return 2 * theta
