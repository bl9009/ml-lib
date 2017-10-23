class LogisticRegressor(object):

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        self.theta = np.ones((1, np_utils.feature_count(X)))

    def predict(self, X):
        pass

    def h(self, x):
        return sigmoid(self.theta, x)

def sigmoid(theta, x):
    z = theta.T.dot(x)

    return 1. / (1 - exp(-z))
