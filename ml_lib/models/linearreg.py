"""Implementation of Linear Regression models."""

import numpy as np

class LinearRegressor(object):
    """Implements a Linear Regression model based on normal equation learningSchedule.

    Attributes:
        theta: Paramaters for linear hypothesis.
    """

    def __init__(self):
        """Initializes Regressor."""
        self.theta = None

    def fit(self, X, y):
        """Fits the model.

        Args:
            X: Training data set.
            y: Labels.
        """
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        h = make_h(self.theta)

        return h(X)


class SgdRegressor(object):
    """Linear Regression model using stochastic gradient descent for learning.

    Attributes:
        theta: Parameters for linear hypothesis.
    """

    def __init__(self, eta0=0.01, annealing=0.25, epochs=100):
        """Initializes Regressor with hyperparameters.

        Args:
            eta0: Starting learning rate.
            annealing: Rate for annealing the learning rate.
            epochs: Number of epochs used for training.
        """
        self.eta0 = eta0
        self.annealing = annealing
        self.epochs = epochs

        self.theta = None

    def fit(self, X, y):
        """Fits the model.

        Args:
            X: Training data set.
            y: Labels.
        """
        X = insert_x0(X)

        self.theta = np.ones((1, feature_count(X)))

        m = instance_count(X)

        gradient_vector = make_mse_gradient_vector(X, y)

        for epoch in range(self.epochs):
            for i in range(m):
                eta = self.__learning_schedule(epoch * m + i + 1)

                self.theta = self.theta - eta * gradient_vector(self.theta)

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        X = insert_x0(X)

        h = make_h(self.theta)

        return h(X)

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

def make_mse(X, y):
    """Closure that returns MSE cost function.

    Args:
        X: Feature set.
        y: Labels.

    Returns:
        MSE function.
    """
    def mse(theta):
        """Calculates MSE based on parameters theta.

        Args:
            theta: Paramaters to calculate MSE for.

        Returns:
            The MSE of given X, y and theta.
        """
        m = instance_count(X)

        h = make_h(theta)

        return (1./(2 * m)) * sum([(h(x_i).item() - y_i)**2 for x_i, y_i in zip(X, y)])

    return mse

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
            Vector of gradients.
        """
        m = instance_count(X)

        index = np.random.randint(m)

        x_i = X[index:index+1]
        y_i = y[index:index+1]

        return 2./m * x_i.T.dot(x_i.dot(theta.T) - y_i).T

    return mse_gradient_vector


def insert_x0(X):
    """Prepend feature x0 = 1 to feature set X.
    Args:

        X: Feature set.

    Returns:
        Updated feature set with prepended feature x0 = 1
    """
    instances, features = X.shape

    features += 1

    tmp_X = np.ones((instances, features))

    tmp_X[0:, 1:] = X

    return tmp_X

def instance_count(X):
    """Return number of instances of feature set."""
    return X.shape[0]

def feature_count(X):
    """Return number of features of feature set."""
    return X.shape[1]
