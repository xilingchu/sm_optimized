import numpy as np
from functools import partial
from autograd import grad

class Adam(object):
    def __init__(self, func, bounds, lr = 0.001, beta1 = 0.9, beta2 = 0.999):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter  = 1
        self.func  = func
        self.bounds = bounds
        # self.grad  = grad(func)
        self.m     = None
        self.v     = None

    def update(self, val):
        if self.m is None:
            self.m = np.zeros(len(val))

        if self.v is None:
            self.v = np.zeros(len(val))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter)/(1.0 - self.beta1**self.iter)

        # grads = self.grad(val)
        # Get the grad
        grads = np.zeros(len(val))
        bratio = np.zeros(len(val))
        for i in range(len(val)):
            val_lb    = np.copy(val)
            val_rb    = np.copy(val)
            val_lb[i] = max(val_lb[i]-self.lr, self.bounds[i,0])
            val_rb[i] = min(val_rb[i]+self.lr, self.bounds[i,1])
            grads[i]  = (self.func(val_rb) - self.func(val_lb))/(val_rb[i] - val_lb[i])
            # Judge if touch the boundary
            if grads[i] > 0:
                dire = 1
            if grads[i] < 0:
                dire = -1
            if grads[i] == 0:
                dire = 0

            if dire == 1:
                dbound = val[i] - val_lb[i]
                if dbound < self.lr:
                    bratio[i] = dbound/self.lr * np.tanh(self.iter)
                else:
                    bratio[i] = 1
            if dire == -1:
                dbound = val_rb[i] - val[i]
                if dbound < self.lr:
                    bratio[i] = dbound/self.lr * np.tanh(self.iter)
                else:
                    bratio[i] = 1
            if dire == 0:
                bratio[i] = 0
            # print(self.func(val_rb), self.func(val_lb))
        print(grads)

        for i in range(len(val)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            val[i]    -= lr_t * bratio[i] * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)
            if val[i] < self.bounds[i, 0]:
                val[i] = self.bounds[i, 0]
            if val[i] > self.bounds[i, 1]:
                val[i] = self.bounds[i, 1]

        return val
