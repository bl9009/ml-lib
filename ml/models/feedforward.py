"""Implementations of feed-forward Artifical Neural Network models."""

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
        deltas = self.__zeros()

        for x_i, y_i in zip(X, y):

            activations = list([x_i])

            # forwardpropagation
            for layer in self.network:
                z = layer.T.dot(tools.insert_intercept(activations[-1]).T)

                activations.append(self.activation(z))

            # backpropagation
            errors = self.__compute_errors(activations, y_i)
            deltas = self.__update_deltas(deltas, activations, errors)

        return self.__compute_gradients(deltas, m=tools.instance_count(X))

    def __compute_errors(self, activations, y_i):
        """Compute errors for each node in each layer."""
        errors = list()

        # compute delta(L)
        errors.append(activations[-1] - y_i)

        # compute delta(L-1), delta(L-2), ... delta(1), delta(0)
        for layer, a in zip(self.network[-2::-1],
                            activations[-2::-1]):
            error = layer.T.dot(errors[0]) * (a * (1 - a))

            errors.insert(0, error)

        return errors

    def __update_deltas(self, deltas, activations, errors):
        """Update deltas used for gradient computation."""
        for l, delta in enumerate(deltas):
            deltas[l] = delta + errors[l+1].dot(activations[l].T)

        return deltas

    def __compute_gradients(self, deltas, m):
        """Compute gradient for each node in each layer."""
        gradients = self.__zeros()

        for l, layer in enumerate(self.network):
            for j, theta in enumerate(layer):
                reg = self.lambda_ * theta[l][j] if j != 0 else 0

                gradients[l][j] = 1 / m * (deltas[l][j] + reg)

        return gradients

    def __zeros(self):
        """Helper to return zero network."""
        return [np.zeros(shape=layer.shape)
                for layer
                in self.network]


def sigmoid(z):
    """Sigmoid activation function."""
    return 1. / (1 + np.exp(-z))

def relu(z):
    """Rectified Linear Unit (ReLU) activation function."""
    return max(0, z)

def tanh(z):
    """tanh activation function."""
    return np.tanh(z)
