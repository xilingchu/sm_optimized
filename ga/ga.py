from krg_optimized.functool.utils import randomRange
from functools import partial
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools

class ga_op(object):
    def __init__(self, func, bound, weights=(1,)):
        self.func   = func
        self.bound  = bound
        self._ndims = bound.shape[0]
        # Create new class
        if not hasattr(creator, 'Fitness'):
            creator.create('Fitness', base.Fitness, weights=tuple(weights))
        else:
            creator.Fitness.weights = weights
        if not hasattr(creator, 'Individual'):
            creator.create('Individual', np.ndarray, fitness=creator.Fitness)
        self._renew_seq()
        self.TB     = base.Toolbox()
        self.TB.register("evaluate",           func)
        self.TB.register("mate",   tools.cxSimulatedBinaryBounded, eta=1, low=list(bound[:,0]), up=list(bound[:,1]))
        self.TB.register("mutate", tools.mutPolynomialBounded, low=list(bound[:,0]), up=list(bound[:,1]), indpb=0.05)
        self.TB.register("select", tools.selTournament, tournsize=3)
        self.TB.register('Individual', tools.initIterate, creator.Individual, self.renew_getseq)
        self.TB.register('Population', tools.initRepeat, list, self.TB.Individual)

    def _get_sequence(self, addi=[]) -> np.ndarray:
        _list = []
        _len_add = 0
        if addi != []:
            _len_add = len(addi)
        for i in range(self._ndims-_len_add):
            _list.append(randomRange(self.bound[i,0], self.bound[i,1]))
        if addi != []:
            _list.append(addi)
        _list = np.array(_list)
        return _list

    def _renew_seq(self, addi=[]):
        self.renew_getseq = partial(self._get_sequence, addi)

    @ staticmethod
    def cxTwoPointCopy(ind1, ind2):
        size = len(ind1)
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else: # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
            = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

        return ind1, ind2

