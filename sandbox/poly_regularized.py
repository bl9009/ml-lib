import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from ml_lib.models.linearreg import SgdRegressor, LinearRegressor
import ml_lib.prep.scaler as scaler
import ml_lib.utils.metrics as metrics

def genTestData(instances=100):
    np.random.seed(0)

    X = np.random.rand(instances, 1)

    X.sort(axis=0)

    y = (np.cos(1.5 * np.pi * X)) + np.random.randn(instances, 1) * 0.1

    #y = 2* X ** 3 + 3 * X**2 + 4 * X + 2 + np.random.randn(instances, 1)

    return X, y

def poly(X, deg=200):
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)

    return poly_features.fit_transform(X)

if __name__ == '__main__':

    X, y = genTestData(30)

    X_test = np.linspace(0, 1, 100)

    X_poly = poly(X, deg=200)
    X_test_poly = poly(X_test.reshape(-1,1), deg=200)

    print(X_poly.shape)

    reg = LinearRegressor(alpha=0.00001)
    reg.fit(X_poly, y)

    y_pred = reg.predict(X_test_poly)

    plt.figure()
    plt.scatter(X, y)
    plt.plot(X_test, y_pred, 'b-')
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.show()
