import matplotlib.pyplot as plt
import numpy as np

class plotSurface(object):
    def __init__(self, func, bounds, n):
        self.func   = func
        self.n      = n
        self.xbound = bounds[:, 0]
        self.ybound = bounds[:, 1]
        # Generate The Mesh to Plot
        self.xplot  = np.linspace(self.xbound[0], self.xbound[1], n)
        self.yplot  = np.linspace(self.ybound[0], self.ybound[1], n)
        self.xyplot = np.zeros([n, n])
        # Get the values of the function
        for i in range(n):
            for j in range(n):
                self.xyplot[i, j] = func(self.xplot[i], self.yplot[j])
        self.fig = plt.figure()

    def clear(self):
        self.fig.clf()
        
    def save(self,path):
        self.fig.savefig(path)

    def contourf(self):
        self.fig.contourf(self.xplot,self.yplot,self.xyplot,50,cmap='rainbow')
        
    def scatter(self, x, y, color='black'):
        self.fig.scatter(x, y, color=color)

    def line(self, x, y, color='black'):
        self.fig.plot(x, y, color=color)
