"""Implementations of feed-forward Artifical Neural Network models."""

import math

import numpy as np

from ..utils import tools

class FeedForwardNN(object):
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
            tbd
        """
        # init Deltas
        Deltas = [np.zeros(shape=layer.shape) 
                 for layer
                 in self.network]        

        for x_i, y_i in zip(X, y):
            deltas = list()

            activations = list([x_i])

            # forward propagation
            for layer in self.network:
                z = layer.T.dot(tools.insert_intercept(activations[-1]).T)

                activations.append(self.activation(z))
                
            # start backpropagation
            # compute delta(L)
            deltas.append(activations[-1] - y_i)

            # compute delta(L-1), delta(L-2), ... delta(1), delta(0)
            for layer, a in zip(self.network[-2::-1],
                                activations[-2::-1]):
                delta = layer.T.dot(deltas[0]) * (a * (1 - a))

                deltas.prepend(delta)

            # update Deltas
            for l, Delta in enumerate(Deltas):
                Delta[l] = Delta + deltas[l+1].dot(activations[l].T)
        

def sigmoid(z):
    """Sigmoid activation function."""
    return 1. / (1 + np.exp(-z))

def relu(z):
    """Rectified Linear Unit (ReLU) activation function."""
    return max(0, z)

def tanh(z):
    """tanh activation function."""
    return math.tanh(z)