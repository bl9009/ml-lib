"""Functions for feature scaling."""

import numpy as np

def normalize(X):
    """Normalize features X (min-max-scaling).

    Args:
        X: Feature set to scale.

    Returns:
        Min-max-scaled feature set.
    """
    min = X.min(axis=0)
    max = X.max(axis=0)

    return (X - min) / (max - min)

def standardize(X):
    """Standardize feature set X.

    Args:
        X: Feature set to scale.

    Returns:
        Scaled feature set.
    """
    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)

    return (X - mean) / variance