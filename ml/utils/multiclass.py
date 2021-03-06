"""Provides mutliclass classification strategies."""

import copy

import numpy as np

from ..utils import tools

POSITIVE_CLASS = 1
NEGATIVE_CLASS = -1

class OneVsAll(object):
    """Multiclass classification implemented using OneVsAll strategy."""

    def __init__(self, clf=None):
        """Initialize multiclass classifier.

        Args:
            clf: Classifier object that will be trained for each class.
        """
        self.clf = clf

        self.trained_clf = dict()
        self.classes = None

    def fit(self, X, y):
        """Train the multiclass classifier for each class in y.

        Args:
            X: Training data set.
            y: Label set.
        """
        self.classes = identify_classes(y)

        for c in self.classes:
            y_aligned = align_labels(y, class_=c)

            tmp_clf = copy.deepcopy(self.clf)

            self.trained_clf[c] = tmp_clf.fit(X, y_aligned)

    def predict(self, X):
        """Make predictions based on trained multiclass classifier.

        Args:
            X: Feature set to predict on.
        """
        results = np.zeros(tools.instance_count(X),
                           len(self.trained_clf))

        for c, clf in self.trained_clf.items():
            results[:, c] = clf.predict(X)

        return evaluate(results, self.trained_clf.keys())


def identify_classes(y):
    """Identifies unique classes in y."""
    return np.unique(y)

def align_labels(y, class_):
    """Adjust set of labels y to suit OneVsAll strategy. Replaces all labels
    that are not equal class_ by an arbitrary value NEGATIVE_CLASS.
    """
    y_aligned = np.copy(y)

    y_aligned[y_aligned == class_] = POSITIVE_CLASS
    y_aligned[y_aligned != class_] = NEGATIVE_CLASS

    return y_aligned

def evaluate(results, classes):
    """Evaluates dict of predicted results and returns an array with
    the final result.

    Args:
        results: A dict with predicted results per class.
        classes: Array of classes.
    """
    return classes[np.where(results[:, :] == POSITIVE_CLASS)[1]]
