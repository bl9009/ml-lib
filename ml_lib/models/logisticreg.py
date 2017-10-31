import numpy as np

from .utils import numpy_utils as np_utils

from sgdreg import SgdRegressor

class LogisticRegressor(SgdRegressor):

    def h(self, x):        
        z = theta.T.dot(x) 

        return sigmoid(z)


def sigmoid(z):
    return 1. / (1 + np.exp(-z))