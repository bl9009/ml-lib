"""Provides Decision Tree models using different learning algorithms."""

import numpy as np

from ..utils import numpy_utils as np_utils
from ..utils import metrics

class DecisionTreeClassifier(object):
    """Decision Tree classification model that is trained with the CART
    algorithm."""

    def __init__(self, max_depth=5):
        """Initialize Decision Tree model."""
        self.tree = BinaryTree()
        self.depth = 0

        self.max_depth = max_depth

    def fit(self, X, y):
        """Fits the Decision Tree model.

        Args:
            X: Training data set.
            y: Labels.
        """
        self.tree.root, self.depth = self._grow_tree(X, y)

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        results = np.ndarray(np_utils.instance_count(X))

        for i, x_i in enumerate(X):
            label = self._evaluate(x_i)

            results[i] = label

    def _grow_tree(self, X, y, depth=0):
        """Builds the decision tree by recursively finding the best split.

        In parallel it returns the depth of the current node, so it is not
        necessary to determine the depth recursively afterwards.

        Args:
            X: Data set to split.
            y: Labels.

        Returns:
            A BinaryTree.Node object representing one node within the
            decision tree.

            The depth of the returned node.
        """
        node_left, node_right = None, None
        depth_left, depth_right = 0, 0

        gini = metrics.gini(X, y)

        node = None

        if gini != 0. and depth < self.max_depth:
            split = self._find_best_split(X, y)

            if np_utils.instance_count(split.left_X) > 1:
                node_left, depth_left = self._grow_tree(split.left_X,
                                                        split.left_y,
                                                        depth+1)

            if np_utils.instance_count(split.right_X) > 1:
                node_right, depth_right = self._grow_tree(split.right_X,
                                                          split.right_y,
                                                          depth+1)

            node = BinaryTree.Node(split.feature_id,
                                   split.threshold,
                                   label=np.argmax(np_utils.label_counts(y)))

            node.left = node_left
            node.right = node_right

        else:
            node = BinaryTree.Node(label=np.argmax(np_utils.label_counts(y)))

        return node, max(depth_left, depth_right, depth)

    def _find_best_split(self, X, y):
        """Splits the dataset X such that both subsets have the best gini
        impurity.

        Args:
            X: Dataset to split.
            y: Labels to split.

        Returns:
            A tuple of arrays representing the splitted subsets of features and
            labels.
        """
        best_J = 0.

        best_split = None

        m = np_utils.instance_count(X)

        for feature, _ in enumerate(X.T):
            for instance in X:
                split = self._split(X,
                                    y,
                                    feature_id=feature,
                                    threshold=instance[feature])

                m_left = np_utils.instance_count(split.left_X)
                m_right = np_utils.instance_count(split.right_X)

                gini_left = metrics.gini(split.left_X, split.left_y)
                gini_right = metrics.gini(split.right_X, split.right_y)

                J = (m_left / m) * gini_left + (m_right / m) * gini_right

                if J <= best_J:
                    best_J = J

                    best_split = split

        return best_split

    def _split(self, X, y, feature_id, threshold):
        """Split the data set X by evaluating feature_id over threshold.

        Args:
            X: Training data set to split.
            feature_id: Feature used to split the set.
            threshold: Threshold used to split the set.

        Returns:
            Two splitted data sets as numpy arrays.
        """
        class Split(object):
            """Class to store splits."""
            def __init__(self,
                         left_X=None,
                         left_y=None,
                         right_X=None,
                         right_y=None,
                         feature_id=0,
                         threshold=0.):
                self.left_X = left_X
                self.left_y = left_y
                self.right_X = right_X
                self.right_y = right_y
                self.feature_id = feature_id
                self.threshold = threshold

        split = Split(feature_id=feature_id, threshold=threshold)

        mask = X[:, feature_id] <= threshold

        split.left_X = X[mask]
        split.left_y = y[mask]

        split.right_X = X[np.logical_not(mask)]
        split.right_y = y[np.logical_not(mask)]

        return split

    def _evaluate(self, x):
        """Evaluate feature vector x against the decision tree.

        Args:
            x: Numpy array with features.
        """
        current = self.tree.root

        while not current.is_leaf():
            if x[current.feature_id] <= current.threshold:
                current = current.left
            else:
                current = current.right


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

        def __eq__(self, other):
            """Check if object is equal to other.

            Args:
                other: Object to check.
            """
            return self.__dict__ == other.__dict__

    def __init__(self):
        """Initialize binary tree."""
        self.root = self.Node()
