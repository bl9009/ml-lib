import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from ml_lib.models.linearreg import SgdRegressor
import ml_lib.prep.scaler as scaler
import ml_lib.utils.metrics as metrics

def genTestData(instances=100):
    X = 6 * np.random.rand(instances, 1) - 3

    X.sort(axis=0)

    y = X ** 3 + 0.5 * X**2 + 0.5 * X + 2 + np.random.randn(instances, 1)

    return X, y

def poly(X, deg=200):
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)

    return poly_features.fit_transform(X)

if __name__ == '__main__':

    X, y = genTestData(10)

    #X_test = 

    X_poly = poly(X, deg=7)

    print(X_poly.shape)

    reg = LinearRegression(normalize=False)
    reg.fit(X_poly, y)
        
    y_pred = reg.predict(X_poly)

    plt.figure()
    plt.plot(X, y, 'rx')
    plt.plot(X, y_pred, 'b-')
    plt.show()