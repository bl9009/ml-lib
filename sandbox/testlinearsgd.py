from linearsgd import LinearSGDRegressor
from scaler import Normalizer

import numpy as np
import matplotlib.pyplot as plot

if __name__ == '__main__':

    reg = LinearSGDRegressor(epochs=2000, eta0=0.1)
    scaler = Normalizer()

    X = 2 * np.random.rand(1000, 1)
    y = 4 + 2 * X #+ np.random.randn(1000, 1)

    features = np.array([[0], [2]])

    X_scaled = scaler.transform(X)
    features_scaled = scaler.transform(features)

    reg.fit(X, y)

    print(reg.theta)

    predicted = reg.predict(features)


    reg.fit(X_scaled, y)

    print(reg.theta)

    predicted_scaled = reg.predict(features_scaled)

    print()

    print(predicted)
    print(predicted_scaled)

    plot.plot(X, y, "b.", label="training")
    plot.plot(features, predicted, "r-", label="labels")
    
    plot.plot(X_scaled, y, "g.", label="training scaled")
    plot.plot(features_scaled, predicted_scaled, "y-", label="predicted scaled")

    plot.show()

    #print(features)