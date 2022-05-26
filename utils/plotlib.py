import matplotlib.pyplot as plt
import numpy as np

class plotOper(object):
    def __init__(self):
        self.figlist = []

    def __enter__(self):
        # Automatically generate the figure
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_all_fig()

    def create_fig(self, fig_name, **kwargs):
        setattr(self, fig_name, plt.figure(**kwargs))

    def clear_fig(self, fig_name):
        fig = getattr(self, fig_name)
        fig.clf()
        delattr(self, fig_name)
        self.figlist.remove(fig_name)

    def clear_all_fig(self):
        for _fig in self.figlist:
            self.clear_fig(_fig)

    def create_ax(self, fig_name, ax_name, pos, **kwargs):
        fig = getattr(self, fig_name)
        setattr(self, ax_name, fig.add_subplot(pos, **kwargs))

    def clear_ax(self, ax_name):
        ax = getattr(self, ax_name)
        ax.cla()
        delattr(self, ax_name)
        
    def save(self, fig_name, path):
        fig = getattr(self, fig_name)
        fig.savefig(path)

class plotSurface(plotOper):
    def __init__(self, func, bounds, n):
        super().__init__()
        self.func   = func
        self.n      = n
        self.xbound = bounds[0, :]
        self.ybound = bounds[1, :]
        self._mesh2d = False
        self._mesh3d = False

    def _mesh_gene(self):
        self.xplot  = np.linspace(self.xbound[0], self.xbound[1], self.n)
        self.yplot  = np.linspace(self.ybound[0], self.ybound[1], self.n)
        self.xyplot = np.zeros([self.n, self.n])
        # Get the values of the function
        for i in range(self.n):
            for j in range(self.n):
                self.xyplot[j, i] = self.func(np.array([self.yplot[i], self.xplot[j]]))
        self._mesh2d = True

    def _mesh_gene3d(self):
        if not self._mesh2d:
            self._mesh_gene()
        self.xplots, self.yplots = np.meshgrid(self.xplot, self.yplot)
        
    def contourf(self, ax_name, **kwargs):
        if not self._mesh2d:
            self._mesh_gene()
        ax = getattr(self, ax_name)
        return ax.contourf(self.xplot,self.yplot,self.xyplot, **kwargs)
        
    def contour(self, ax_name, **kwargs):
        if not self._mesh2d:
            self._mesh_gene()
        ax = getattr(self, ax_name)
        return ax.contour(self.xplot,self.yplot,self.xyplot, **kwargs)

    def scatter(self, ax_name, *args, **kwargs):
        ax = getattr(self, ax_name)
        ax.scatter(*args, **kwargs)

    def line(self, ax_name, x, y, **kwargs):
        ax = getattr(self, ax_name)
        ax.plot(x, y, **kwargs)

    def line3d(self, ax_name, x, y, z, **kwargs):
        ax = getattr(self, ax_name)
        ax.plot3D(x, y, z, **kwargs)
        
    def colorbar(self, fig_name, var):
        fig = getattr(self, fig_name)
        fig.colorbar(var)

    def surface3d(self, ax_name, **kwargs):
        if not self._mesh3d:
            self._mesh_gene3d()
        ax = getattr(self, ax_name)
        ax.plot_surface(self.xplots, self.yplots, self.xyplot, **kwargs)
