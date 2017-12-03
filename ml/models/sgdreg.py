"""Super class for SGD regression models."""

import abc

import numpy as np

from ..utils import tools

class SgdRegressor(abc.ABC):
    """Abstract super class for SGD regression models.

    Attributes:
        theta: Parameters for linear hypothesis.
    """
    def __init__(
            self,
            eta0=0.01,
            annealing=0.25,
            epochs=100,
            alpha=0.,
            l1_ratio=1.):
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

        self.theta = None

    def fit(self, X, y):
        """Fits the model using stochastic gradient descent algorithm.

        Args:
            X: Training data set.
            y: Labels.
        """
        X = tools.insert_intercept(X)

        self.theta = np.ones((1, tools.feature_count(X)))

        m = tools.instance_count(X)

        gradient_vector = self.__make_loss_gradient_vector(X, y)

        for epoch in range(self.epochs):
            for i in range(m):
                eta = self.__learning_schedule(epoch * m + i + 1)

                l1_penalty = self.alpha * lasso_vector(self.theta)
                l2_penalty = self.alpha * ridge_vector(self.theta)

                penalty = self.l1_ratio * l1_penalty + (1. - self.l1_ratio) / 2. * l2_penalty

                self.theta = self.theta - eta * gradient_vector(self) + penalty

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        X = tools.insert_intercept(X)

        return self.h(X)

    @abc.abstractmethod
    def h(self, x):
        """Hypothesis function h(x) used to make predictions.

        Args:
            x: Vector of features to make prediction of.

        Returns:
            Vector of predictions.
        """
        pass

    def __learning_schedule(self, t):
        """Adjust learning rate depending on performed iterations t.

        Args:
            t: Number of iterations performed.

        Returns:
            Adjusted learning rate eta.
        """
        return self.eta0 / t**self.annealing

    def __make_loss_gradient_vector(self, X, y):
        """Closure that returns a function to calculate the loss gradient vector.

        Args:
            X: Feature set.
            y: Labels.

        Returns:
            Function to calculate gradient vector of MSE function.
        """
        def loss_gradient_vector(self):
            """Calculates the gradient vector of loss function.

            Args:
                theta: Parameters to calculate MSE gradient for.

            Returns:
                Vector of gradients as numpy array.
            """
            m = tools.instance_count(X)

            index = np.random.randint(m)

            x_i = X[index:index+1]
            y_i = y[index:index+1]

            return 2 * x_i.T.dot(self.h(x_i) - y_i).T

        return loss_gradient_vector

def lasso_vector(theta):
    """Calculates gradient vector for LASSO regularization.

    Args:
        theta: Parameters to calculate gradient vector for.

    Returns:
        Numpy array with gradients.
    """
    return sign(theta)

@tools.vectorize
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
