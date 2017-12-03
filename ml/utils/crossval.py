"""Cross validation functionality."""

import numpy as np

from . import tools

class CrossValidation(object):
    """This class allows cross validation of machine learning models."""

    def __init__(self,
                 classifier=None,
                 val_ratio=0.2,
                 folds=1,
                 seed=42):
        """Initializes cross validation.

        Args:
            classifier: The classifier to cross validate.
            val_ratio: The ratio of validation data to be used for CV.
            fold: Number of runs (folds) of cross validation to be performed.
            seed: Seed for randomness.
        """
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
        """Computes the cross validation Score for each fold, including:
            - accuracy
            - true positives
            - true negatives
            - false positives
            - false negatives
            - precision and
            - recall.

        Args:
            X: Complete training data set.
            y: Complete training label set.
        """
        for _ in range(self.folds):
            train_X, train_y, val_X, val_y = self._generate_split(X, y)

            self.classifier.fit(train_X, train_y)

            predictions = self.classifier.predict(val_X)

            self.accuracy.append(accuracy(predictions, val_y))
            self.tp.append(true_positives(predictions, val_y))
            self.tn.append(true_negatives(predictions, val_y))
            self.fp.append(false_positives(predictions, val_y))
            self.fn.append(false_negatives(predictions, val_y))
            self.precision.append(precision(predictions, val_y))
            self.recall.append(recall(predictions, val_y))

    def _generate_split(self, X, y):
        """Generates a random split of training data into training and
        validation data by given ratio.

        Args:
            X: Complete training data set.
            y: Complete training label set.
        """
        m = tools.instance_count(X)

        m_val = int(round(m * self.val_ratio))

        np.random.seed(self.seed)

        val_ids = np.arange(m_val)
        np.random.shuffle(val_ids)

        mask = np.ones(len(X), dtype=bool)
        mask[val_ids] = False

        val_X = X[val_ids, :]
        val_y = y[val_ids, :]

        train_X = X[mask, :]
        train_y = y[mask, :]

        return train_X, train_y, val_X, val_y

def accuracy(predicted_y, validation_y):
    """Computes the accuracy."""
    m = tools.instance_count(validation_y)

    acc = (predicted_y == validation_y).sum() / m

    return acc

def true_positives(predicted_y, validation_y):
    """Retrieves number of true positives."""
    val_pos = np.where(validation_y == 1)
    pred_pos = np.where(predicted_y == 1)

    return np.in1d(pred_pos, val_pos).sum()

def true_negatives(predicted_y, validation_y):
    """Retrieves number of true negatives."""
    val_neg = np.where(validation_y == 0)
    pred_neg = np.where(predicted_y == 0)

    return np.in1d(pred_neg, val_neg).sum()

def false_positives(predicted_y, validation_y):
    """Retrieves number of false positives."""
    val_neg = np.where(validation_y == 0)
    pred_pos = np.where(predicted_y == 1)

    return np.in1d(pred_pos, val_neg).sum()

def false_negatives(predicted_y, validation_y):
    """Retrievs number of false negatives."""
    val_pos = np.where(validation_y == 1)
    pred_neg = np.where(predicted_y == 0)

    return np.in1d(pred_neg, val_pos).sum()

def precision(predicted_y, validation_y):
    """Computes the precision, meaning how many of predicted positives
    are actual positives? (also called TPR, True Positive Rate)
    """
    true_pos = true_positives(predicted_y, validation_y)
    false_pos = false_positives(predicted_y, validation_y)

    return true_pos / (true_pos + false_pos)

def recall(predicted_y, validation_y):
    """Computes the recall, meaning how many of all actual positives
    have been predicted correctly?
    """
    true_pos = true_positives(predicted_y, validation_y)
    false_neg = false_negatives(predicted_y, validation_y)

    return true_pos / (true_pos + false_neg)
