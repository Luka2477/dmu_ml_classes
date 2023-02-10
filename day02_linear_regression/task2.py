import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)

    plt.plot(X, y, "b.")
    plt.axis([0, 2, 0, 15])

    plt.plot()
    plt.show()
