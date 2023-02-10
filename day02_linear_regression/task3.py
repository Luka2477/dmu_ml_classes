import sys

import matplotlib.pyplot as plt
import numpy as np


def cost(a, b, _x, _y):
    # Evaluate half MSE (Mean square error)
    m = len(_y)
    error = a + b * _x - _y
    J = np.sum(error ** 2) / (2 * m)
    return J


if __name__ == '__main__':
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)

    aInterval = np.arange(3, 4, 0.001)
    bInterval = np.arange(3, 4, 0.001)

    minCost = (0, 0, sys.maxsize)
    for aTheta in aInterval:
        for bTheta in bInterval:
            temp = cost(aTheta, bTheta, x, y)
            if temp < minCost[2]:
                minCost = (aTheta, bTheta, temp)

    print(minCost)
