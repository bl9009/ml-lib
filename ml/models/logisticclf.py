"""Logistic classification model."""

from logisticreg import LogisticRegression

class LogisticClassifier(LogisticRegression):

    def __init__(
            self,
            threshold=0.5,
            eta0=0.01,
            annealing=0.25,
            epochs=100,
            alpha=0.,
            l1_ratio=1.):
        self.threshold = threshold

        super(LogisticClassifier, self).__init__(
            eta0,
            annealing,
            epochs,
            alpha,
            l1_ratio)

    def predict(self, X):
        result = super(LogisticClassifier, self).predict(X)

        return result > self.threshold
