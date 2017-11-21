import unittest

import numpy as np

from ml.models.treeclf import DecisionTreeClassifier
from ml.models.treeclf import BinaryTree
from ml.utils import metrics

class TestDecisionTreeClassifier(unittest.TestCase):

    def test_gini(self):
        X, y, gini = data_set()

        self.assertEqual(metrics.gini(X, y), gini)

    def test_split(self):
        X, y, _ = data_set()

        test_split = data_set_splitted_1_5()

        test_left_X, test_left_y, test_right_X, test_right_y = test_split

        clf = DecisionTreeClassifier()

        split = clf._split(X, y, feature_id=1, threshold=5)

        self.assertArraysEqual(split.left_X, test_left_X)
        self.assertArraysEqual(split.left_y, test_left_y)
        self.assertArraysEqual(split.right_X, test_right_X)
        self.assertArraysEqual(split.right_y, test_right_y)

    def test_grow_tree(self):
        X = np.array([[12, 34, 62],
                      [3, 86, 28],
                      [42, 18, 81],
                      [23, 52, 21],
                      [14, 16, 42]])

        y = np.array([1, 1, 0, 0, 1])

    def assertArraysEqual(self, X1, X2):
        # workaround to assert numpy arrays
        self.assertTrue((X1 == X2).all())


class TestBinaryTree(unittest.TestCase):

    def test_eq(self):
        node1 = BinaryTree.Node()
        node2 = BinaryTree.Node()
        node3 = BinaryTree.Node(feature_id=1)

        self.assertTrue(node1 == node2)
        self.assertTrue(node1 != node3)

    def test_add_node(self):
        tree = BinaryTree()

        left_node = BinaryTree.Node(feature_id=1)
        right_node = BinaryTree.Node(feature_id=5)

        tree.root.set_left(left_node)
        tree.root.set_right(right_node)

        self.assertEqual(tree.root.left, left_node)
        self.assertEqual(tree.root.right, right_node)

    def test_is_leaf(self):
        tree = BinaryTree()

        node1 = BinaryTree.Node()
        node2 = BinaryTree.Node()

        tree.root.set_left(node1)
        tree.root.left.set_left(node2)

        self.assertFalse(tree.root.left.is_leaf())
        self.assertTrue(tree.root.left.left.is_leaf())


def data_set():
    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [3, 5, 7],
                  [8, 1, 2]])

    #y = np.array([[0],
    #             [1],
    #             [1],
    #             [0],
    #             [1]])

    y = np.array([0, 1, 1, 0, 1])

    gini = 0.48

    return X, y, gini

def data_set_splitted_1_5():
    # data set splitted at feature 1 and threshold 5
    left_X = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [3, 5, 7],
                       [8, 1, 2]])

    right_X = np.array([[7, 8, 9]])

    left_y = np.array([0, 1, 0, 1])

    right_y = np.array([1])

    return left_X, left_y, right_X, right_y

def data_set_splitted_0_8():
    # data set splitted at feature 0 and threshold 8
    pass

def data_set_splitted_2_1():
    # data set splitted at feature 2 and threshold 1
    pass


def generate_data(m, n, seed):
    np.random.seed(seed)

    X = np.random.randint(0, 100, size=(m, n))
    y = np.random.randint(0, 2, size=(m, 1))

    return X, y

if __name__ == '__main__':
    unittest.main()
