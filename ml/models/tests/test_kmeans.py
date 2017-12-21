"""Tests for K-Means model."""

import unittest

import numpy as np

from ml.models.kmeans import KMeans

class TestKMeans(unittest.TestCase):
    """Tests for KMeans class."""

    class MockKMeans(KMeans):
        """Mock class KMeans to access protected methods."""

        def init_means(self, X):
            """Exhibit protected _initial_means method."""
            self._init_means(X)

        def assign_clusters(self, X):
            """Exhibit protected method _assign_clusters."""
            return self._assign_clusters(X)

    def test_init_means(self):
        """Test initial means selection."""
        k = 3
        instances = 100

        X = np.random.randint(100, size=(instances, 10))

        np.random.seed(42)

        means = X[np.random.randint(instances, size=k), :]

        model = self.MockKMeans(k, seed=42)

        model.init_means(X)

        test_means = model.means

        self.assertTrue(np.array_equal(means, test_means))

    def test_assign_clusters(self):
        """Test cluster assignment."""
        k = 2

        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9],
                      [1, 5, 1],
                      [8, 5, 6],
                      [3, 1, 0]])

        means = np.array([[7, 8, 9],
                          [1, 2, 3]])

        model = self.MockKMeans(k)
        model.means = means

        clusters = np.array([1, 0, 0, 1, 0, 1])

        test_clusters = model.assign_clusters(X)

        self.assertTrue(np.array_equal(clusters, test_clusters))
