"""Provides mutliclass classification strategies."""

import numpy as np

class OneVsAll(object):
    """Multiclass classification implemented using OneVsAll strategy."""

    def __init__(self, classifier=None):
        """Initialize multiclass classifier.

        Args:
            classifier: Classifier object that will be trained for each class.
        """
        self.classifier = classifier

    def fit(self, X, y):
        """Train the multiclass classifier for each class in y.

        Args:
            X: Training data set.
            y: Label set.
        """
        classes = identify_classes(y)

    def predict(self, X):
        """Make predictions based on trained multiclass classifier.

        Args:
            X: Feature set to predict on.
        """
        pass


def identify_classes(y):
    """Identifies unique classes in y."""
    return np.unique(y)
