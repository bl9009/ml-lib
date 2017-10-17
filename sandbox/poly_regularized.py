import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from ml_lib.models.linearreg import SgdRegressor

def genTestData(instances=100):
    X = 6 * np.random.rand(instances, 1) - 3
    y = 0.5 * X**2 + X + 2 + np.random.randn(instances, 1)

    return X, y

def poly(X, deg=200):
    poly_features = PolynomialFeatures(degree=deg, include_bias=False)

    return poly_features.fit_transform(X)

if __name__ == '__main__':

    X, y = genTestData()

    X_poly = poly(X, deg=10)

    reg = SgdRegressor(epochs=10000, eta0=0.00001)

    reg.fit(X_poly, y)

    y_pred = reg.predict(X_poly)

    print(reg.theta)

    plt.plot(X, y, 'rx')
    plt.plot(X, y_pred, 'bo')
    plt.show()