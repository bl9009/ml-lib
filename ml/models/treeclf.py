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
        self.tree.root = self.__grow_tree(X, y)

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

    def __grow_tree(self, X, y):
        """Builds the decision tree by recursively finding the best split.

        Args:
            X: Data set to split.
            y: Labels.

        Returns:
            A BinaryTree.Node object representing one node within the
            decision tree.
        """
        best_gini = 0.

        best_feature_id = 0

        best_threshold = 0.

        best_left_X = None
        best_left_y = None

        best_right_X = None
        best_right_y = None

        for j, in enumerate(X.T):
            for instance in X:
                split = self.__split(X,
                                     y,
                                     feature_id=j,
                                     threshold=instance[j])

                left_X, left_y, right_X, right_y = split

                gini_left = self.__gini(left_X, left_y)
                gini_right = self.__gini(right_X, right_y)

                if gini_left > best_gini or gini_right > best_gini:
                    best_feature_id = j

                    best_threshold = instance[j]

                    best_gini = max(gini_left, gini_right)

                    best_left_X = left_X
                    best_left_y = left_y

                    best_right_X = left_X
                    best_right_y = left_y

        node = BinaryTree.Node(best_feature_id,
                               best_threshold,
                               best_gini)

        node.left = self.__grow_tree(best_left_X, best_left_y)
        node.right = self.__grow_tree(best_right_X, best_right_y)

        return node

    def _split(self, X, y, feature_id, threshold):
        """Split the data set X by evaluating feature_id over threshold.

        Args:
            X: Training data set to split.
            feature_id: Feature used to split the set.
            threshold: Threshold used to split the set.

        Returns:
            Two splitted data sets as numpy arrays.
        """
        mask = X[:, feature_id] <= threshold

        left_X = X[mask]
        left_y = y[mask]

        right_X = X[np.logical_not(mask)]
        right_y = y[np.logical_not(mask)]

        return left_X, left_y, right_X, right_y

    def _gini(self, X, y):
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

            # decision trees grow top down, so _no_ nodes
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
        self.root = Node()

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
