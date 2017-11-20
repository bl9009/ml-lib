import unittest

import numpy as np

from ml.models.treeclf import DecisionTreeClassifier
from ml.models.treeclf import BinaryTree

class TestDecisionTreeClassifier(unittest.TestCase):

    def test_gini(self):
        X, y, gini = data_set()

        clf = DecisionTreeClassifier()

        self.assertEqual(clf._gini(X, y), gini)

    def test_split(self):
        X, y, _ = data_set()

        test_left_X, test_left_y, test_right_X, test_right_y = data_set_splitted_1_5()

        clf = DecisionTreeClassifier()

        left_X, left_y, right_X, right_y = clf._split(X, y, feature_id=1, threshold=5)

        self.assertArraysEqual(left_X, test_left_X)
        self.assertArraysEqual(left_y, test_left_y)
        self.assertArraysEqual(right_X, test_right_X)
        self.assertArraysEqual(right_y, test_right_y)

    def test_grow_tree(self):
        pass

    def assertArraysEqual(self, X1, X2):
        # workaround to assert numpy arrays
        self.assertTrue((X1 == X2).all())


class TestBinaryTree(unittest.TestCase):

    def test_add_node(self):
        tree = BinaryTree()

        tree.root = Node()

    def test_is_leaf(self):
        pass

    def test_find(self):
        pass

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
