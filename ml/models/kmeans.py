"""K-Means clustering model."""

import numpy as np

from ..utils import tools

class KMeans(object):
    """K-Means clustering model implementation based on Lloyd's algorithm."""

    def __init__(self, k=2, seed=42):
        """Initialize K-Means model.
        
        Args:
            k: Number of clusters.
            seed: Seed used to randomly initialize means.
        """
        self.k = k

        np.random.seed(seed)

    def fit(self, X, y=None):
        """Compute clustering using Lloyd's algorithm.

        Args:
            X: Training data set.
            y: Irrelevant.
        """
        _ = y # workaround in order to keep argument y and maintain interface

        finished = False

        clusters = None

        means = self._initial_means(X)

        while finished is False:
            new_clusters = self._assign_clusters(X, means)

            if clusters == new_clusters:
                finished = True
            else:
                clusters = new_clusters

                means = self._compute_means(X)

        return clusters

    def predict(self, X):
        """Determine cluster for each instance in dataset X.

        Args:
            X: Dataset.

        Returns:
            A numpy array containing the predicted values.
        """
        pass

    def _initial_means(self, X):
        """Randomly select initial means."""
        instances = tools.instance_count(X)

        return X[np.random.randint(instances, size=self.k), :]

    def _assign_clusters(self, X, means):
        """Assign instances of X to clusters depending on distance to means."""
        pass

    def _compute_means(self, X):
        """Compute set of k means."""
        pass
