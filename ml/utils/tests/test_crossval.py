import unittest

import numpy as np

from ml.utils import crossval
from ml.utils.crossval import CrossValidation

class TestCrossValidation(unittest.TestCase):

    def test_score(self):
        pass

    def test_generate_split(self):
        cv = CrossValidation()

        split = cv._generate_split(X, y)

        test_train_X, test_train_y, test_val_X, test_val_y = split

    def test_accuracy(self):
        pred_y, val_y = y_data()

        test_acc = 0.75

        acc = crossval.accuracy(pred_y, val_y)

        self.assertEqual(acc, test_acc)

    def test_true_positives(self):
        pred_y, val_y = y_data()

        test_tp = 5

        tp = crossval.true_positives(pred_y, val_y)

        self.assertEqual(tp, test_tp)

    def test_true_negatives(self):        
        pred_y, val_y = y_data()

        test_tn = 4

        tn = crossval.true_negatives(pred_y, val_y)

        self.assertEqual(tn, test_tn)

    def test_false_positives(self):
        pred_y, val_y = y_data()

        test_fp = 1

        fp = crossval.false_positives(pred_y, val_y)

        self.assertEqual(fp, test_fp)

    def test_false_negatives(self):
        pred_y, val_y = y_data()

        test_fn = 2

        fn = crossval.false_negatives(pred_y, val_y)

        self.assertEqual(fn, test_fn)

    def test_precision(self):
        pred_y, val_y = y_data()

        test_prec = 5 / 6

        prec = crossval.precision(pred_y, val_y)

        self.assertEqual(prec, test_prec)

    def test_recall(self):
        pred_y, val_y = y_data()

        test_rec = 5 / 7

        rec = crossval.recall(pred_y, val_y)

        self.assertEqual(rec, test_rec)


def y_data():
    predicted  = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0])
    validation = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

    return predicted, validation