"""Implementations of feed-forward Artifical Neural Network models."""

import math

import numpy as np

from ..utils import tools

class FeedForwardNN(object):
    """Feed-forward Multi-Layer Perceptron model."""

    def __init__(self,
                 hidden_nodes=tuple(),
                 activation=None,
                 lambda_=0.,
                 seed=42):
        """Initialize the Neural Netowrk.

        Args:
            hidden_nodes: Tuple with number of nodes for each hidden layer.
            activation: Reference to activation function.
        """
        self.hidden_nodes = hidden_nodes
        self.activation = activation
        self.lambda_ = lambda_
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
            A numpy array containing the probability for each class.
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
        a = X

        for layer in self.network:
            z = layer.T.dot(tools.insert_intercept(a).T)

            a = self.activation(z).T

        return a

    def _back_propagate(self, X, y):
        """Calculate derivatives of log loss function J using
        back propagation algorithm.

        Args:
            X: Training data set.
            y: Training labels.

        Returns:
            List of numpy arrays containing gradient for
            each node in the network.
        """
        # init Deltas and gradients
        def zeros():
            """Helper to return zero network."""
            return [np.zeros(shape=layer.shape)
                    for layer
                    in self.network]

        deltas = zeros()
        gradients = zeros()

        for x_i, y_i in zip(X, y):
            errors = list()

            activations = list([x_i])

            # forward propagation
            for layer in self.network:
                z = layer.T.dot(tools.insert_intercept(activations[-1]).T)

                activations.append(self.activation(z))

            # start backpropagation
            # compute delta(L)
            errors.append(activations[-1] - y_i)

            # compute delta(L-1), delta(L-2), ... delta(1), delta(0)
            for layer, a in zip(self.network[-2::-1],
                                activations[-2::-1]):
                error = layer.T.dot(errors[0]) * (a * (1 - a))

                errors.insert(0, error)

            # update Deltas
            for l, delta in enumerate(deltas):
                delta[l] = delta + errors[l+1].dot(activations[l].T)

        # compute gradients
        m = tools.instance_count(X)

        for l, layer in enumerate(self.network):
            for j, theta in enumerate(layer):
                reg = self.lambda_ * theta[l][j] if j != 0 else 0

                gradients[l][j] = 1 / m * (deltas[l][j] + reg)

        return gradients


def sigmoid(z):
    """Sigmoid activation function."""
    return 1. / (1 + np.exp(-z))

def relu(z):
    """Rectified Linear Unit (ReLU) activation function."""
    return max(0, z)

def tanh(z):
    """tanh activation function."""
    return math.tanh(z)
