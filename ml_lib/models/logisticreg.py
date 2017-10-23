class LogisticRegressor(object):

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def h(self, x):
        return sigmoid(self.theta, x)

def sigmoid(theta, x):
    z = theta.T.dot(x)

    return 1. / (1 - exp(-z))
