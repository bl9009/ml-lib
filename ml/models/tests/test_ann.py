"""Tests for ANN module."""

import unittest

import numpy as np

from ml.models.ann import ANN

class TestANN(unittest.TestCase):

    def test_build_network(self):
        test_network = [
            np.random.randn(10, 4),
            np.random.randn(4, 6),
            np.random.randn(6, 3)]

        ann = ANN(hidden_nodes=(4, 6))

        n = 10
        k = 3

        ann._build_network(n, k)

        network = ann.network

        for layer, test_layer in zip(network, test_network):
            self.assertEqual(layer.shape, test_layer.shape)