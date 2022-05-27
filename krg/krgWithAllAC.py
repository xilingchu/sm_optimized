import numpy as np
from scipy.stats import norm
from smt.surrogate_models import KRG


class krg(object):

    def __init__(self, X: np.ndarray, Y: np.ndarray, _iter) -> None:
        self.X = X
        self.Y = Y
        self.k = (1 * np.log(len(Y) * (_iter + 1)**2 * np.pi**2 /
                             (6 * 0.1)))**0.5  # 置信区间递增公式
        self.zeta = 0.1  # 用于为EI和PI添加扰动
        self.sm = KRG(theta0=[1e-4], print_global=False)
        self.sm.set_training_values(X, Y)
        self.sm.train()

    def EI(self, X: np.ndarray) -> np.ndarray:   # 改良期望采集
        _f_min = np.min(self.Y)
        _pred = self.sm.predict_values(np.atleast_2d(X))
        _sig = np.sqrt(self.sm.predict_variances(np.atleast_2d(X)))
        _args0 = (_f_min - _pred) / _sig
        _args1 = (_f_min - _pred) * norm.cdf(_args0)
        _args2 = _sig * norm.pdf(_args0)
        if _sig.size == 1 and _sig == 0.0:  # can be use only if one point is computed
            return np.array(0.0)
        ei = -_args1 - _args2  # 方便优化库求解（约定优化最小值问题）
        # ei = -ei
        return ei

    def PI(self, X: np.ndarray) -> np.ndarray:  # 改良概率采集
        _f_min = np.min(self.Y)
        _pred = self.sm.predict_values(np.atleast_2d(X))
        _sig = np.sqrt(self.sm.predict_variances(np.atleast_2d(X)))
        _args0 = (_f_min - _pred + self.zeta) / _sig
        _args1 = norm.cdf(_args0)
        if _sig.size == 1 and _sig == 0.0:
            return np.array(0.0)
        Pi = -_args1  # 方便优化库求解（约定优化最小值问题）
        return Pi

    def LCB(self, X: np.ndarray) -> np.ndarray:  # 下置信边界采集
        _pred = self.sm.predict_values(np.atleast_2d(X))
        # 标准差
        _sig = np.sqrt(self.sm.predict_variances(np.atleast_2d(X)))
        LC = _pred - self.k * _sig
        return LC

    def sm_ga(self, X: np.ndarray) -> np.ndarray:
        y = self.sm.predict_values(np.atleast_2d(X))
        return y
