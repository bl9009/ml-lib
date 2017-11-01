"""Functions for feature scaling."""

def normalize(X):
    """Normalize features X (min-max-scaling).

    Args:
        X: Feature set to scale.

    Returns:
        Min-max-scaled feature set.
    """

    return (X - X.min()) / (X.max() - X.min())
