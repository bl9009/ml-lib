"""Implementations of feed-forward Artifical Neural Network models."""

import numpy as np

from ..utils import tools

class MLP(object):
    """Feed-forward Multi-Layer Perceptron model."""

    def __init__(self,
                 hidden_nodes=tuple(),
                 activation=None,
                 seed=42):
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
        return self._feed_forward(X)

    def _build_network(self, n, k):
        """Construct the network.

        Args:
            n: number of features (input nodes).
            k: number of classes (output nodes).
        """
        nodes = (n,) + self.hidden_nodes + (k,)

        np.random.seed(self.seed)

        self.network = [np.random.randn(i + 1, j)
                        for i, j
                        in zip(nodes[:-1], nodes[1:])]

    def _feed_forward(self, X):
        """Feed forward network with dataset X."""
        out = X

        for layer in self.network:
            z = layer.T.dot(tools.insert_intercept(out).T)

            out = self.activation(z).T

        return out


def sigmoid(z):
    return z

def relu(z):
    return z

def tanh(z):
    return z

def identity(z):
    return z