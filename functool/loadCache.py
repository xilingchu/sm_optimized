import numpy as np
from os import chdir
# import pickle


def loadcache(Dir):
    chdir(Dir)
    BEST_X = np.load('BEST_X.npy')
    BEST_Y = np.load('BEST_Y.npy')
    MAX_EI_X = list(np.load('MAX_EI_X.npy'))
    MAX_EI_Y = np.load('MAX_EI_Y.npy')

    return BEST_X, BEST_Y, MAX_EI_X, MAX_EI_Y


def loadcache1(Dir):
    DATA = np.load(Dir + 'data.npz')
    BEST_Y = DATA['BEST_Y']
    REAL_BEST_Y = DATA['REAL_BEST_Y']
    ITER = DATA['ITER']
    return BEST_Y, REAL_BEST_Y, ITER
