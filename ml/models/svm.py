"""Implementation of Support Vector Machine and corresponding Kernels."""

import abc

import numpy as np

class SVM(object):
    """SVM classification model."""

    def __init__(self, kernel=LinearKernel(), C=1.0):
        """Initializes classifier with hyperparameters.
        
        Args:
            kernel: Instance of a kernel.
            C: Soft margin penalty.
        """
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        """Fits the model parameters.

        Args:
            X: Training data set.
            y: Labels.
        """
        pass

    def predict(self, X):
        """Performs predictions based on fitted model.

        Args:
            X: Feature set.

        Returns:
            A numpy array containing the predicted values.
        """
        pass


class Kernel(abc.ABC):
    """Abstract base class for SVM kernels."""

    @abc.abstractmethod
    def K(self, a, b):
        """Kernel method to be overridden in inheriting class.

        Args:
            a, b: Vectors of features.

        Returns:
            Transformed feature vector.
        """
        pass

class LinearKernel(Kernel):
    """Linear kernel."""

    def K(self, a, b):
        """Computes the kernel transformation.

        Args:
            a, b: Vectors of features.

        Returns:
            Transformed feature vector.
        """
        return a.T.dot(b)

class PolyKernel(Kernel):
    """Polynomial kernel."""

    def __init__(self, degree=2, gamma=1., r=0.):
        """Initialize kernel.

        Args:
            degree: Degree of polynomial transformation.
            gamma: Kernel coefficient.
            r: Independent kernel term.
        """
        self.degree = degree
        self.gamma = gamma
        self.r = r

    def K(self, a, b):
        """Computes the kernel transformation.

        Args:
            a, b: Vectors of features.

        Returns:
            Transformed feature vector.
        """
        return np.power((self.gamma * a.T.dot(b) + self.r), self.degree)

class RBFKernel(Kernel):
    """Gaussian RBF kernel."""

    def __init__(self, gamma=1.):
        """Initialize kernel.

        Args:
            gamma: Kernel coefficient.
        """
        self.gamma = gamma

    def K(self, a, b):
        """Computes the kernel transformation.

        Args:
            a, b: Vectors of features.

        Returns:
            Transformed feature vector.
        """
        return np.exp(- self.gamma * np.linalg.norm(a - b))

class Sigmoid(Kernel):
    """Sigmoid kernel."""

    def __init__(self, gamma=1., r=0.):
        """Initialize kernel.

        Args:
            gamma: Kernel coefficient.
            r: Independent kernel term.
        """
        self.gamma = gamma
        self.r = r

    def K(self, a, b):
        """Computes the kernel transformation.

        Args:
            a, b: Vectors of features.

        Returns:
            Transformed feature vector.
        """
        return np.tanh(self.gamma * a.T.dot(b) + self.r)
