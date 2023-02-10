import matplotlib.pyplot as plt
import numpy as np


def linear_regression(_x, _y, _theta0=0, _theta1=0, epochs=10000, learning_rate=0.0001):
    _cost = 0
    N = float(len(_y))
    for i in range(epochs):
        y_hypothesis = (_theta1 * _x) + _theta0
        _cost = sum([data ** 2 for data in (_y - y_hypothesis)]) / N
        theta1_gradient = -(2 / N) * sum(_x * (_y - y_hypothesis))
        theta0_gradient = -(2 / N) * sum(_y - y_hypothesis)
        _theta0 = _theta0 - (learning_rate * theta0_gradient)
        _theta1 = _theta1 - (learning_rate * theta1_gradient)

    return _theta0, _theta1, _cost


if __name__ == '__main__':
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)
    
    theta0, theta1, cost = linear_regression(x, y)

    plt.plot(x, y, "b.")
    plt.axis([0, 2, 0, 12])

    # let's plot that line:
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot([theta0, theta1])
    plt.plot(X_new, y_predict, "g-")

    plt.show()
