"""Tests for K-Means model."""

import unittest

import numpy as np

from ml.models.kmeans import KMeans

class TestKMeans(unittest.TestCase):
    """Tests for KMeans class."""

    class MockKMeans(KMeans):
        """Mock class KMeans to access protected methods."""

        def initial_means(self, X):
            """Exhibit protected _initial_means method."""
            return self._initial_means(X)

    def test_initial_means(self):
        """Test initial means selection."""
        k = 3
        instances = 100

        X = np.random.randint(100, size=(instances, 10))

        np.random.seed(42)

        means = X[np.random.randint(instances, size=k), :]

        model = self.MockKMeans(k, seed=42)
        
        test_means = model.initial_means(X)

        self.assertTrue(np.array_equal(means, test_means))