"""Cross validation functionality."""

import numpy as np

from . import numpy_utils as np_utils

class CrossValidation(object):

    def __init__(self,
                 classifier=None,
                 val_ratio=0.2,
                 folds=1,
                 seed=42):
        self.classifier = classifier
        self.val_ratio = val_ratio
        self.folds = folds
        self.seed = seed

        self.accuracy = list()
        self.tp = list()
        self.tn = list()
        self.fp = list()
        self.fn = list()
        self.precision = list()
        self.recall = list()

    def score(self, X, y):
        for fold in range(folds):
            train_X, train_y, val_X, val_y = self._generate_split(X, y)

            fitted_clf = self.classifier.fit(train_X, train_y)

            predictions = self.classifier.predict(val_X)

            self.accuracy.append(accuracy(predictions, val_y))
            self.tp.append(true_positives(predictions, val_y))
            self.tn.append(true_negatives(predictions, val_y))
            self.fp.append(false_positives(predictions, val_y))
            self.fn.append(false_negatives(predictions, val_y))
            self.precision.append(precision(predictions, val_y))
            self.recall.append(recall(predictions, val_y))

    def _generate_split(self, X, y):
        m = np_utils.instance_count(X)
        n = np_utils.feature_count(X)

        m_val = int(round(m * self.val_ratio))
        m_train = m - m_val

        np.random.seed(self.seed)

        val_ids = np.arange(m_val)
        numpy.random.shuffle(val_ids)

        mask = np.ones(len(X), dtype=bool)
        mask[val_ids] = False

        val_X = X[val_ids, :]
        val_y = y[val_ids, :]

        train_X = X[mask, :]
        train_y = y[mask, :]

        return train_X, train_y, val_X, val_y

def accuracy(predicted_y, validation_y):
    m = np_utils.instance_count(validation_y)

    acc = (predicted_y == validation_y).sum() / m

    return acc

def true_positives(predicted_y, validation_y):
    val_pos = np.where(validation_y == 1)
    pred_pos = np.where(predicted_y == 1)

    return np.in1d(pred_pos, val_pos).sum()

def true_negatives(predicted_y, validation_y):
    val_neg = np.where(validation_y == 0)
    pred_neg = np.where(predicted_y == 0)

    return np.in1d(pred_neg, val_neg).sum()

def false_positives(predicted_y, validation_y):
    val_neg = np.where(validation_y == 0)
    pred_pos = np.where(predicted_y == 1)

    return np.in1d(pred_pos, val_neg).sum()

def false_negatives(predicted_y, validation_y):
    val_pos = np.where(validation_y == 1)
    pred_neg = np.where(predicted_y == 0)

    return np.in1d(pred_neg, val_pos).sum()

def precision(predicted_y, validation_y):
    # how many of predicted positives are actual positives? TPR
    true_pos = true_positives(predicted_y, validation_y)
    false_pos = false_positives(predicted_y, validation_y)

    return true_pos / (true_pos + false_pos)

def recall(predicted_y, validation_y):
    # how many of all actual positives have been predicted correctly?
    true_pos = true_positives(predicted_y, validation_y)
    false_neg = false_negatives(predicted_y, validation_y)

    return true_pos / (true_pos + false_neg)
