"""Functions and closures for MSE calculation"""

def make_mse(X, y):
    """Closure that returns MSE cost function.

    Args:
        X: Feature set.
        y: Labels.

    Returns:
        MSE function.
    """
    def mse(theta):
        """Calculates MSE based on parameters theta.

        Args:
            theta: Paramaters to calculate MSE for.

        Returns:
            The MSE of given X, y and theta.
        """
        m = instance_count(X)

        h = make_h(theta)

        return (1./(2 * m)) * sum([(h(x_i) - y_i)**2 for x_i, y_i in zip(X, y)])

    return mse
