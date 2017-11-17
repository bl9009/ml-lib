"""Provides Decision Tree models using different learning algorithms."""

import numpy as np

from ..utils import numpy_utils as np_utils

class DecisionTreeClassifier(object):
    """Decision Tree classification model that is trained with the CART
    algorithm."""

    def __init__(self):
        """Initialize Decision Tree model."""
        self.tree = BinaryTree()

    def fit(self, X, y):
        """Fits the Decision Tree model.

        Args:
            X: Training data set.
            y: Labels.
        """
        for j, feature in enumerate(X.T):
            for i, instance in enumerate(X):
                left, right = self.__split(X, i, X[i, j])

                # best_gain = self.__gini(left)
                # best_split = BinaryTree.Node(j, threshold, gini, samples)


    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        results = np.ndarray(np_utils.instance_count(X))

        for i, x in enumerate(X):
            label = self.tree.find(x)

            results[i] = label

    def __split(self, X, feature_id, threshold):
        """Split the data set X by evaluating feature_id over threshold.

        Args:
            X: Training data set to split.
            feature_id: Feature used to split the set.
            threshold: Threshold used to split the set.

        Returns:
            Two splitted data sets as numpy arrays.
        """
        left = [X[:, feature_id] <= threshold]
        right = [X[:, feature_id] > threshold]

        return left, right

    def __gini(self, X, y):
        """Calculate gini impurity of given data set X.

        Args:
            X: Training data set.
            y: Label set

        Returns:
            Gini impurity as float.
        """
        m = np_utils.instance_count(X)

        label_counts = np_utils.label_counts(y)

        return 1 - sum([(n / m)**2 for n in label_counts])




class BinaryTree(object):
    """Binary Tree implementation used for growing a decision tree with CART."""

    class Node(object):
        """Node class for binary trees."""

        def __init__(self, feature_id=0, threshold=0., gini=0., samples=0., value=None, label=""):
            """Initialize binary tree node.

            Args:
                left: Node on the left branch of the node.
                right: Node on the right branch of the node.
                feature_id: Feature selected for this node.
                threshold: Decision boundary threshold.
                gini: Gini impurity metric.
                samples: Number of training samples.
                value: List of counts of each label classified by this node.
                label: Most probable label classified by this node.
            """
            self.left = None
            self.right = None

            self.feature_id = feature_id
            self.threshold = threshold
            self.gini = gini
            self.samples = samples
            self.value = value
            self.label = label

        def set_left(self, node):
            """Set the left child node."""

            # decision trees grow top down, so no nodes
            # will be added in between
            self.left = node

        def set_right(self, node):
            """Set the right child node."""
            self.right = node

        def is_leaf(self):
            """Check if node is a leaf."""
            return self.left is None and self.right is None

    def __init__(self):
        """Initialize binary tree."""
        self.root = None

    def find(self, x):
        """Find node with class fitting given feature vector.

        Args:
            x: Numpy array with features.
        """
        current = self.root

        while not current.is_leaf():
            if x[current.feature_id] <= current.threshold:
                current = current.left
            else:
                current = current.right
