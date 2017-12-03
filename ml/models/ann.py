"""Implementations of feed-forward Artifical Neural Network models."""

import abc

import numpy as np

from ..utils import tools

class FeedForwardANN(abc.ABC):
    """Abstract base class for forwared-fed
    Artificial Neural Network models.
    """

    def __init__(self, hidden_nodes=tuple(), activation=None, seed=42):
        """Initialize the Neural Netowrk.

        Args:
            hidden_nodes: Tuple with number of nodes for each hidden layer.
            activation: Reference to activation function.
        """
        self.hidden_nodes = hidden_nodes
        self.activation = activation
        self.seed = seed

        self.network = None

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

    def _build_network(self, n, k):
        """Builds the network.

        Args:
            n: number of features (input nodes).
            k: number of classes (output nodes).
        """
        nodes = (n,) + self.hidden_nodes + (k,)

        np.random.seed(self.seed)

        self.network = [np.random.randn(i, j)
                        for i, j
                        in zip(nodes[:-1], nodes[1:])]

    def _feed_forward(self, X):
        """Feed forward network with dataset X."""
        out = X.T

        for layer in self.network:
            z = layer.T.dot(out)

            out = self.activation(z)

        return out

class MLP(FeedForwardANN):
    """Feed-forward Multi-Layer Perceptron model."""

    def fit(self, X, y):
        """Fits the MLP model.

        Args:
            X: Training data set.
            y: Labels.
        """
        n = tools.feature_count(X)
        k = tools.class_count(y)

        self._build_network(n, k)

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        pass
