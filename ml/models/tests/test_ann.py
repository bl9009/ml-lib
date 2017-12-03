"""Tests for ANN module."""

import unittest

import numpy as np

from ml.models.ann import ANN

class TestANN(unittest.TestCase):
    """Tests for ANN."""

    class MockANN(ANN):
        """Mock class ANN to access protected methods."""

        def build_network(self, n, k):
            """Exhibit protected _build_network method."""
            self._build_network(n, k)

    def test_build_network(self):
        """Test network construction."""

        test_network = [
            np.random.randn(10, 4),
            np.random.randn(4, 6),
            np.random.randn(6, 3)]

        ann = self.MockANN(hidden_nodes=(4, 6))

        n = 10
        k = 3

        ann.build_network(n, k)

        network = ann.network

        for layer, test_layer in zip(network, test_network):
            self.assertEqual(layer.shape, test_layer.shape)
