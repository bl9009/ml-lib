"""Implementations of Artifical Neural Network models."""

import abc

import numpy as np

class ANN(abc.ABC):
    """Abstract base class for Artificial Neural Network models."""

    def __init__(self, hidden_layout=tuple(), activation=None):
        """Initialize the Neural Netowrk.

        Args:
            hidden_layout: Tuple representing the layout of hidden layers.
            activation: Reference to activation function.
        """
        self.hidden_layout = hidden_layout
        self.activation = activation

        self.theta = None

    def fit(self, X, y):
        """Fits the ANN model.

        Args:
            X: Training data set.
            y: Labels.
        """
        pass

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        pass


class MLP(ANN):
    """Feed-forward Multi-layer Perceptron model."""

    def fit(self, X, y):
        """Fits the MLP model.

        Args:
            X: Training data set.
            y: Labels.
        """
        pass

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        pass