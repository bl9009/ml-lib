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

def accuracy(predicted_y, validation_y):
    pass

def true_positives(predicted_y, validation_y):
    pass

def true_negatives(predicted_y, validation_y):
    pass

def false_positives(predicted_y, validation_y):
    pass

def false_negatives(predicted_y, validation_y):
    pass

def precision(predicted_y, validation_y):
    pass

def recall(predicted_y, validation_y):
    pass
