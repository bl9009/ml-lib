import unittest

import numpy as np

from ml.utils import crossval
from ml.utils.crossval import CrossValidation

class TestCrossValidation(unittest.TestCase):

    def test_score(self):
        pass

    def test_generate_split(self):
        pass

    def test_accuracy(self):
        pred_y, val_y = y_data()

        recall = crossval.accuracy(pred_y, val_y)

    def test_true_positives(self):
        pass

    def test_true_negatives(self):
        pass

    def test_false_positives(self):
        pass

    def test_false_negatives(self):
        pass

    def test_precision(self):
        pass

    def test_recall(self):
        pass


def y_data():
    predicted = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0])
    validation = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

    return predicted, validation