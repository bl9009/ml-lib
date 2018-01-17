"""Tests for MLP module."""

import unittest

import numpy as np

from ml.models import feedforward
from ml.models.feedforward import FeedForwardNN

class TestFeedForwardNN(unittest.TestCase):
    """Tests for ANN."""

    class MockFF(FeedForwardNN):
        """Inherit class FeedForwardNN to access protected methods."""

        def build_network(self, n, k):
            """Exhibit protected _build_network method."""
            self._build_network(n, k)

        def feed_forward(self, X):
            """Exhibit protected _feed_forward method."""
            return self._feed_forward(X)

        def compute_activations(self, x_i):
            """Exhibit private __compute_activations method."""
            return self._FeedForwardNN__compute_activations(x_i)

    def test_build_network(self):
        """Test network construction."""
        test_network = [
            np.random.randn(4, 11),
            np.random.randn(6, 5),
            np.random.randn(3, 7)]

        ann = self.MockFF(hidden_nodes=(4, 6))

        n = 10
        k = 3

        ann.build_network(n, k)

        network = ann.network

        for layer, test_layer in zip(network, test_network):
            self.assertEqual(layer.shape, test_layer.shape)

    def test_feed_forward(self):
        """Test forward feeding."""
        ann = self.MockFF(hidden_nodes=(4, 6), activation=feedforward.tanh)

        ann.build_network(2, 3)

        X = np.random.randint(0, 100, size=(130, 2))

        out = ann.feed_forward(X)

        self.assertEqual(out.shape, (130, 3))

    def test_compute_activations(self):
        """Test computation of activations."""
        ann = self.MockFF(hidden_nodes=(4,), activation=feedforward.relu)

        ann.build_network(4, 3)

        x_i = np.array([4, 7, 8, 4])

        activations = ann.compute_activations(x_i)

        activations_test = [
            np.array([[4, 7, 8, 4]]),
            np.array([[15.72510207, 9.8692025, 0., 0.]]),
            np.array([[0., 0., 28.39235005]])
            ]

        for test, actual in zip(activations_test, activations):
            self.assertTrue(
                np.array_equal(test.round(decimals=5),
                               actual.round(decimals=5)))

    def test_compute_errors(self):
        """Test compute errors helper function."""
        #ann = self.MockFF(hidden_nodes=(4, 6), activation=feedforward.relu)

        #ann.build_network(2, 3)
        pass
