import numpy as np
# 将求值功能扩展到大于0的偶数维度


def rosenbrock(X: np.ndarray) -> np.ndarray:
    # The Rosenbrock Function:
    # y(X) = \sum_{i=1}^{n} (1 - X(i-1))^2 + 100 * (X(i) - X(i-1))^2)
    try:
        X.shape[1]
    except:
        X = np.atleast_2d([X])

    if X.shape[1] == 1:
        raise Exception('The length of the X array should higher than 2!')
    else:
        _len = X.shape[1]

    y = np.zeros(X.shape[0])
    for i in range(0, _len, 2):
        y += (1 - X[:, i])**2 + 100 * (X[:, i + 1] - X[:, i]**2)**2
    return y
