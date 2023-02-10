import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D


def cost(a, b):
    # Evaluate half MSE (Mean square error)
    m = len(yDots)
    error = a + b * xDots - yDots
    J = np.sum(error ** 2) / (2 * m)
    return J


if __name__ == '__main__':
    xDots = 2 * np.random.rand(100, 1)
    yDots = -5 + 7 * xDots + np.random.randn(100, 1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    aInterval = np.arange(-10, 10, 0.05)
    bInterval = np.arange(-10, 10, 0.05)

    X, Y = np.meshgrid(aInterval, bInterval)
    zs = np.array([cost(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Cost')
    plt.show()
