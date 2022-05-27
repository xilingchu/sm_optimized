import numpy as np


def threeHumpCamel(X: np.ndarray) -> np.ndarray:
    try:
        X.shape[1]
    except:
        X = np.atleast_2d([X])

    if X.shape[1] == 1:
        raise Exception('The length of the X array should higher than 2!')
    else:
        _len = X.shape[1]

    y = 2 * X[:, 0]**2 - 1.05 * X[:, 0]**4 + (
        1 / 6) * X[:, 0]**6 + X[:, 0] * X[:, 1] + X[:, 1]**2
    # 有3个局部最小值，全局最小为【0，0】-0
    y = np.zeros(X.shape[0])
    for i in range(0, _len, 2):
        y += 2 * X[:, i]**2 - 1.05 * X[:, i]**4 + (
            1 / 6) * X[:, i]**6 + X[:, i] * X[:, i+1] + X[:, i+1]**2
    return y
