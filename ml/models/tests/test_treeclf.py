"""Tests for the treeclf module."""

import unittest

import numpy as np

from ml.models.treeclf import DecisionTreeClassifier
from ml.models.treeclf import BinaryTree
from ml.utils import metrics

class TestDecisionTreeClassifier(unittest.TestCase):
    """Tests for DecisionTreeClassifier."""

    class MockDecisionTreeClassifier(DecisionTreeClassifier):
        """Mock class DecisionTreeClassifier to access protected methods."""

        def split(self, X, y, feature_id, threshold):
            """Exhibit protected _split method."""
            return self._split(X, y, feature_id, threshold)

        def find_best_split(self, X, y):
            """Exhibit protected _find_best_split method."""
            return self._find_best_split(X, y)

    def test_gini(self):
        """Test gini computation."""
        X, y, gini = data_set()

        self.assertEqual(metrics.gini(X, y), gini)

    def test_split(self):
        """Test split of data set."""
        X, y, _ = data_set()

        test_split = data_set_splitted_1_18()

        test_left_X, test_left_y, test_right_X, test_right_y = test_split

        clf = self.MockDecisionTreeClassifier()

        split = clf.split(X, y, feature_id=1, threshold=18)

        self.assertArraysEqual(split.left_X, test_left_X)
        self.assertArraysEqual(split.left_y, test_left_y)
        self.assertArraysEqual(split.right_X, test_right_X)
        self.assertArraysEqual(split.right_y, test_right_y)

    def test_best_split(self):
        """Test finding the best split of a data set."""
        X, y, _ = data_set()

        test_split = data_set_best_split()

        test_left_X, test_left_y, test_right_X, test_right_y = test_split

        clf = self.MockDecisionTreeClassifier()

        split = clf.find_best_split(X, y)

        self.assertArraysEqual(split.left_X, test_left_X)
        self.assertArraysEqual(split.left_y, test_left_y)
        self.assertArraysEqual(split.right_X, test_right_X)
        self.assertArraysEqual(split.right_y, test_right_y)

    def test_grow_tree(self):
        """Test growing the decision tree."""
        pass

    def assertArraysEqual(self, X1, X2):
        """Workaround to assert numpy arrays."""
        self.assertTrue((X1 == X2).all())


class TestBinaryTree(unittest.TestCase):
    """Test for the BinaryTree."""

    def test_eq(self):
        """Test equality operator."""
        node1 = BinaryTree.Node()
        node2 = BinaryTree.Node()
        node3 = BinaryTree.Node(feature_id=1)

        self.assertTrue(node1 == node2)
        self.assertTrue(node1 != node3)

    def test_add_node(self):
        """Test adding a left and right node to a node."""
        tree = BinaryTree()

        left_node = BinaryTree.Node(feature_id=1)
        right_node = BinaryTree.Node(feature_id=5)

        tree.root.set_left(left_node)
        tree.root.set_right(right_node)

        self.assertEqual(tree.root.left, left_node)
        self.assertEqual(tree.root.right, right_node)

    def test_is_leaf(self):
        """Test if node is a leaf."""
        tree = BinaryTree()

        node1 = BinaryTree.Node()
        node2 = BinaryTree.Node()

        tree.root.set_left(node1)
        tree.root.left.set_left(node2)

        self.assertFalse(tree.root.left.is_leaf())
        self.assertTrue(tree.root.left.left.is_leaf())


def data_set():
    """Return sample data set and its gini impurity."""
    X = np.array([[12, 34, 62],
                  [3, 86, 28],
                  [42, 18, 81],
                  [23, 52, 21],
                  [14, 16, 42]])

    y = np.array([1, 1, 0, 0, 1])

    gini = 0.48

    return X, y, gini

def data_set_splitted_1_18():
    """Return a data set splitted at feature 1 and threshold 18."""
    left_X = np.array([[42, 18, 81],
                       [14, 16, 42]])

    right_X = np.array([[12, 34, 62],
                        [3, 86, 28],
                        [23, 52, 21]])

    left_y = np.array([0, 1])

    right_y = np.array([1, 1, 0])

    return left_X, left_y, right_X, right_y

def data_set_best_split():
    """Return best split of data set."""
    left_X = np.array([[12, 34, 62],
                       [3, 86, 28],
                       [14, 16, 42]])

    right_X = np.array([[42, 18, 81],
                        [23, 52, 21]])

    left_y = np.array([1, 1, 1])

    right_y = np.array([0, 0])

    return left_X, left_y, right_X, right_y

def generate_data(m, n, seed):
    """Generate random data set."""
    np.random.seed(seed)

    X = np.random.randint(0, 100, size=(m, n))
    y = np.random.randint(0, 2, size=(m, 1))

    return X, y

if __name__ == '__main__':
    unittest.main()
