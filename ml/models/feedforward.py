"""Implementations of feed-forward Artifical Neural Network models."""

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
        # 0. init Deltas
        Deltas = [np.zeros(shape=layer.shape) 
                 for layer
                 in self.network]        

        # 1. compute a for every layer (forward prop)
        activations = list()
        zs = list()

        a = X

        for layer in self.network:
            z = layer.T.dot(tools.insert_intercept(a).T)

            zs.append(z)

            a = self.activation(z).T

            activations.append(a)

        # 2. compute delta for last layer
        deltas = list()

        delta = a - y

        deltas.append(delta)

        # 3. compute delta for other layers
        for a, z in zip(activations[-2::-1], zs[-2::-1]):
            delta = deltas[-1] * activation_derivative(z)

            deltas.append(delta)            

        # 4. compute Delta

        # 5. comput D
        


def sigmoid(z):
    return 1. / (1 + np.exp(-z))

def relu(z):
    return z

def tanh(z):
    return z