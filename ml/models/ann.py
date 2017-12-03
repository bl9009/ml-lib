"""Implementations of Artifical Neural Network models."""

import abc

import numpy as np

from ..utils import tools

class ANN(abc.ABC):
    """Abstract base class for Artificial Neural Network models."""

    def __init__(self, hidden_nodes=tuple(), activation=None):
        """Initialize the Neural Netowrk.

        Args:
            hidden_nodes: Tuple with number of nodes for each hidden layer.
            activation: Reference to activation function.
        """
        self.hidden_nodes = hidden_nodes
        self.activation = activation

        self.network = None
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

    def _build_network(self, n, k):
        """Builds the network.

        Args:
            n: number of features (input nodes).
            k: number of classes (output nodes).
        """
        nodes = (n,) + self.hidden_nodes + (k,)

        self.network = [np.random.randn(i, j) for i, j in zip(nodes[:-1], nodes[1:])]

class MLP(ANN):
    """Feed-forward Multi-layer Perceptron model."""

    def fit(self, X, y):
        """Fits the MLP model.

        Args:
            X: Training data set.
            y: Labels.
        """
        n = tools.feature_count(X)
        k = tools.label_counts(y)

        self._build_network(n, k)

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        pass
