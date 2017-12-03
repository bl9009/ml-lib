"""Tests for the crossval module."""

import unittest

import numpy as np

from ml.utils import crossval
from ml.utils.crossval import CrossValidation

class TestCrossValidation(unittest.TestCase):
    """Tests for CrossValidation."""

    class MockCrossValidation(CrossValidation):
        """Mock class CrossValidation to access protected methods."""

        def generate_split(self, X, y):
            """Exhibit protected _generate_split method."""
            return self._generate_split(X, y)

    def test_score(self):
        """Test score computation."""
        pass

    def test_generate_split(self):
        """Test splitting of data."""
        cv = self.MockCrossValidation(val_ratio=0.3)

        X = np.random.randint(0, 100, size=(100, 5))
        y = np.random.randint(0, 1, size=(100, 1))

        split = cv.generate_split(X, y)

        test_train_X, test_train_y, test_val_X, test_val_y = split

        self.assertEqual(len(test_train_X), 70)
        self.assertEqual(len(test_train_y), 70)
        self.assertEqual(len(test_val_X), 30)
        self.assertEqual(len(test_val_y), 30)

    def test_accuracy(self):
        """Test accuracy computation."""
        pred_y, val_y = y_data()

        test_acc = 0.75

        acc = crossval.accuracy(pred_y, val_y)

        self.assertEqual(acc, test_acc)

    def test_true_positives(self):
        """Test retrieval of true positives."""
        pred_y, val_y = y_data()

        test_tp = 5

        tp = crossval.true_positives(pred_y, val_y)

        self.assertEqual(tp, test_tp)

    def test_true_negatives(self):
        """Test retrieval of true negatives."""
        pred_y, val_y = y_data()

        test_tn = 4

        tn = crossval.true_negatives(pred_y, val_y)

        self.assertEqual(tn, test_tn)

    def test_false_positives(self):
        """Test retrieval of false positives."""
        pred_y, val_y = y_data()

        test_fp = 1

        fp = crossval.false_positives(pred_y, val_y)

        self.assertEqual(fp, test_fp)

    def test_false_negatives(self):
        """Test retrieval of false negatives."""
        pred_y, val_y = y_data()

        test_fn = 2

        fn = crossval.false_negatives(pred_y, val_y)

        self.assertEqual(fn, test_fn)

    def test_precision(self):
        """Test precision computation."""
        pred_y, val_y = y_data()

        test_prec = 5 / 6

        prec = crossval.precision(pred_y, val_y)

        self.assertEqual(prec, test_prec)

    def test_recall(self):
        """Test recall computation."""
        pred_y, val_y = y_data()

        test_rec = 5 / 7

        rec = crossval.recall(pred_y, val_y)

        self.assertEqual(rec, test_rec)


def y_data():
    """Return some test labels."""
    predicted = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0])
    validation = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

    return predicted, validation
