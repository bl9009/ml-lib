""" Compare own implementation performance vs SciKit Learn """

import numpy as np

import os
import time

import sklearn.linear_model as skmodels
import ml_lib.models.linearreg as mymodels

def genTestData(instances=1000):
    X = 2 * np.random.rand(instances, 1)
    y = 4 + 2 * X + np.random.randn(instances, 1)

    return X, y

def compareLinearRegressor(X, y, featureScaling=True):
    pass

def compareSgdRegressor(X, y, featureScaling=True):
    print("\tLINEAR SGD\n")

    myReg = mymodels.SgdRegressor()
    skReg = skmodels.SGDRegressor()

    myTrainingDuration = takeTime(myReg.fit, X, y)
    skTrainingDuration = takeTime(skReg.fit, X, y.ravel())

    print("TRAINING DURATION")
    print("my\t: {}".format(myTrainingDuration))
    print("sklearn\t: {}".format(skTrainingDuration))

    print("\nPARAMETERS")
    print("my\t: {}".format(myReg.theta))
    print("sklearn\t: {}".format(skReg.coef_))


def takeTime(func, *args):
    start = time.time()

    func(*args)

    return time.time() - start

def main():
    X, y = genTestData()

    compareSgdRegressor(X, y)

if __name__ == "__main__":
    main()
