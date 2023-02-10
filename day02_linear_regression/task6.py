# import matplotlib.pyplot as plt
# import numpy as np
#
# if __name__ == '__main__':
#     X = 2 * np.random.rand(100, 1)
#     y = 4 + 3 * X + np.random.randn(100, 1)
#
#     X_b = np.c_[np.ones((100, 1)), X]
#     theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
#
#     X_new = np.array([[0], [2]])
#     X_new_b = np.c_[np.ones((2, 1)), X_new]
#     y_predict = X_new_b.dot(theta_best)
#
#     plt.plot(X, y, "g.")
#     plt.axis([0, 2, 0, 15])
#     plt.plot(X_new, y_predict, "r-")
#
#     plt.plot()
#     plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

if __name__ == '__main__':
    X = 2 * np.random.rand(500, 1)
    y = 4 + 3 * X + np.random.randn(500, 1)

    lm = linear_model.LinearRegression()
    model = lm.fit(X, y)

    plt.plot(X, y, "g.")
    plt.axis([0, 2, 0, 15])

    # fit function
    f = lambda x: lm.coef_ * x + lm.intercept_
    plt.plot(X, f(X), c="red")
    plt.plot()
    plt.show()