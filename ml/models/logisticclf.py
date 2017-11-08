"""Logistic classification model."""

from logisticreg import LogisticRegression

class LogisticClassifier(LogisticRegression):
    """Classification model based on Logistic Regression."""

    def __init__(
            self,
            threshold=0.5,
            eta0=0.01,
            annealing=0.25,
            epochs=100,
            alpha=0.,
            l1_ratio=1.):
        """Initializes classifier with hyperparameters.

        Regularizes model with elastic net by setting regularization factor
        alpha. Setting l1_ratio to 1.0 performs l1 regularization (LASSO),
        l1_ratio to 0.0 for l2 regularization (ridge)

        Args:
            threshold: Threshold classification is based on.
            eta0: Starting learning rate.
            annealing: Rate for annealing the learning rate.
            epochs: Number of epochs used for training.
            alpha: Regularization factor
            l1_ratio: l1 penalty ratio
        """
        self.threshold = threshold

        super(LogisticClassifier, self).__init__(
            eta0,
            annealing,
            epochs,
            alpha,
            l1_ratio)

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        result = super(LogisticClassifier, self).predict(X)

        return result > self.threshold
