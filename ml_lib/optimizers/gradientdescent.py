"""Gradient Descent implementations."""

import numpy as np

import numpy_utils as np_utils

class StochasticGradientDescent(object):
    """Stochastic Gradient Descent optimization.

    Attributes:
        X: mxn matrix representing training set where m is the number of
        instances and n is the number of features.
        y: vector with m values as training labels.
    """

    def __init__(self, X, y, regularizer=None):
        self.X = X
        self.y = y

    def do_step(self, model, eta=0.01):
        """Do one step of Stochastic Gradient Descent.

        Args:
            model: A model object to optimize.
            eta: Learning rate.

        Returns:
            Model object with updated parameters theta.
        """
        penalty = regularizer.penalty(model) if self.regularizer not None else 0.

        model.theta = model.theta - eta * gradient_vector(model) + penalty

        return model

    def optimize(self, model, eta=0.01, epochs=100):
        """Optimize model using stochastic gradient descent for given number
        of epochs, so that J(theta) is a minimum.

        Args:
            model: Model object that will be optimized.
            eta: Learning rate.
            epochs: Number of iterations over the training set.

        Returns:
            A model with optimized parmeters theta.
        """
        m = np_utils.instance_count(self.X)

        for epoch in range(self.epochs):
            for i in range(m):
                model = do_step(model, eta)

        return model

    def gradient_vector(self, model):
        """Calculates the gradient vector with a random training instance.

        Args:
            model: Model object with hypothesis h and parameters theta.

        Returns:
            A vector with gradients for each parameter theta.
        """
        m = np_utils.instance_count(X)

        index = np.random.randint(m)

        x_i = self.X[index:index+1]
        y_i = self.y[index:index+1]

        return 2./m * x_i.T.dot(model.h(x_i) - y_i).T
