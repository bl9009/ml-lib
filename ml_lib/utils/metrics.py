"""Functions for error and loss calculation."""

def mse(y, y_predicted):
    """Calculates MSE of given target and predicted values.

    Args:
        y: Ground truth.
        y_predicted: Predictions.

    Returns:
        The MSE.
    """
    m = y.size()

    return (1./(2. * m)) * sum((y_predicted - y) ** 2)

def rss(y, y_predicted):
    """Calculates RSS (residual sum of squares) for given target
    and predicted values.

    Args:
        y: Ground truth.
        y_predicted: Predicted values.

    Returns:
        Residual sum of squares.
    """
    return sum((y_predicted - y) ** 2)
