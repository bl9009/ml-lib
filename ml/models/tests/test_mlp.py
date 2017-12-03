"""Tests for MLP module."""

import unittest

import numpy as np

from ml.models.mlp import MLP

class TestMLP(unittest.TestCase):
    """Tests for ANN."""

    class MockMLP(MLP):
        """Inherit class MLP to access protected methods."""

        def build_network(self, n, k):
            """Exhibit protected _build_network method."""
            self._build_network(n, k)

        def feed_forward(self, X):
            """Exhibit protected _feed_forward method."""
            return self._feed_forward(X)

    def test_build_network(self):
        """Test network construction."""

        test_network = [
            np.random.randn(11, 4),
            np.random.randn(5, 6),
            np.random.randn(7, 3)]

        ann = self.MockMLP(hidden_nodes=(4, 6))

        n = 10
        k = 3

        ann.build_network(n, k)

        network = ann.network

        for layer, test_layer in zip(network, test_network):
            self.assertEqual(layer.shape, test_layer.shape)

    def test_feed_forward(self):
        """Test forward feeding."""

        def activation(z):
            """Activation dummy."""
            return z

        ann = self.MockMLP(hidden_nodes=(4, 6), activation=activation)

        ann.build_network(2, 3)

        X = np.random.randint(0, 100, size=(130, 2))

        out = ann.feed_forward(X)

        self.assertEqual(out.shape, (130, 3))
