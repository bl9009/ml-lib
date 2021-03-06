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

        self.means = None

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

        self._init_means(X)

        while finished is False:
            new_clusters = self._assign_clusters(X)

            if np.array_equal(new_clusters, clusters):
                finished = True
            else:
                clusters = new_clusters

                self._update_means(X, clusters)

    def predict(self, X):
        """Determine cluster for each instance in dataset X.

        Args:
            X: Dataset.

        Returns:
            A numpy array containing the predicted values.
        """
        return self._assign_clusters(X)

    def _init_means(self, X):
        """Randomly select initial means."""
        instances = tools.instance_count(X)

        self.means = X[np.random.randint(instances, size=self.k), :]

    def _assign_clusters(self, X):
        """Assign instances of X to clusters depending on distance to means."""
        clusters = np.full(shape=tools.instance_count(X), fill_value=-1)

        for id_, instance in enumerate(X):
            cluster = np.argmin(np.linalg.norm(instance - self.means, axis=1))

            clusters[id_] = cluster

        return clusters

    def _update_means(self, X, clusters):
        """Compute and update means."""
        self.means = compute_means(X, self.k, clusters)


def compute_means(X, k, clusters):
    """Compute set of k means."""
    means = np.zeros(shape=(k, tools.feature_count(X)))

    for cluster_id in range(k):
        cluster = cluster_array(X, clusters, cluster_id)

        means[cluster_id] = mean_of_cluster(cluster)

    return means

def mean_of_cluster(cluster):
    """Compute the mean of given cluster."""
    return cluster.mean(axis=0)

def cluster_array(X, clusters, cluster_id):
    """Get an array with all instances per cluster."""
    return X[np.where(clusters == cluster_id)]
