import numpy as np


def branin(X: np.ndarray) -> np.ndarray:
    # The Branin Function:
    # $f(\mathbf{x})=a\left(x_{2}-b x_{1}^{2}+c x_{1}-r\right)^{2}+s(1-t) \cos \left(x_{1}\right)+s$
    try:
        X.shape[1]
    except:
        X = np.atleast_2d([X])

    if X.shape[1] == 1:
        raise Exception('The length of the X array should higher than 2!')
    else:
        _len = X.shape[1]

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    # 有3个全局最小值：0.397887：[-pi, 12.275][pi,2.275][9.42478,2.475],极值为 0.397887
    y = np.zeros(X.shape[0])
    for i in range(0, _len, 2):
        y += a * (X[:, i + 1] - b * (X[:, i])**2 + c * X[:, i] -
                  r)**2 + s * (1 - t) * np.cos(X[:, i]) + s
    return y
