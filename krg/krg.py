import numpy as np
from scipy.stats import norm
from smt.surrogate_models import KRG

class krg(object):
    def __init__(self, X:np.ndarray, Y:np.ndarray) -> None:
        self.X = X
        self.Y = Y
        self.sm = KRG(theta0=[1e-4], print_global=False)
        self.sm.set_training_values(X, Y)
        self.sm.train()

    def EI(self, X:np.ndarray) -> np.ndarray:
        _f_min = np.min(self.Y)
        _pred  = self.sm.predict_values(np.atleast_2d(X))
        _sig   = np.sqrt(self.sm.predict_variances(np.atleast_2d(X)))
        _args0 = (_f_min - _pred) / _sig
        _args1 = (_f_min - _pred) * norm.cdf(_args0)
        _args2 = _sig * norm.pdf(_args0)
        if _sig.size == 1 and _sig == 0.0:  # can be use only if one point is computed
            return np.array(0.0)
        ei = _args1 + _args2
        return ei

    def sm_ga(self, X:np.ndarray) -> np.ndarray:
        y  = self.sm.predict_values(np.atleast_2d(X))
        return y
