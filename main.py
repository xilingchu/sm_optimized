from krg_optimized.ga.ga import ga_op
from krg_optimized.functool.rosenbrock import rosenbrock
from krg_optimized.krg.krg import krg
from krg_optimized.adam.adam import Adam
from krg_optimized.utils.plotlib import plotSurface
from deap import tools
from matplotlib.pyplot import xlim
from smt.sampling_methods import LHS
from scipy.optimize import minimize, Bounds, fmin_l_bfgs_b
import numpy as np
import random
import matplotlib.pyplot as plt
import autograd.numpy as np2
from matplotlib import cm
from pathlib import Path

def main_ga(func, bound:np.ndarray, pop_num:int, iter_min:int, iter_max:int, weights=(1,), addi:list=[]):
    CXPB, MUTPB = 0.9, 0.05
    _ga  = ga_op(func, bound, weights)
    if addi != []:
        _ga._renew_seq(addi)
    _pop = _ga.TB.Population(n=pop_num)

    #------------ Start Evolution ------------#
    print('Start Evolution!')
    _fitnesses = list(map(_ga.TB.evaluate, _pop))
    for ind, fit in zip(_pop, _fitnesses):
        ind.fitness.values = fit

    _fits = [ind.fitness.values[0] for ind in _pop]

    _iter = 0

    eps = 1.0 
    best_value = []

    while not(_iter >= iter_max or eps < 1e-6):
        _iter += 1
        print('Generation %i'%_iter)
        # Select the next generation
        offspring = _ga.TB.select(_pop, pop_num)
        # Clone the selected individuals
        offspring = list(map(_ga.TB.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
        # cross two individuals with probability CXPB
            if random.random() < CXPB:
                _ga.TB.mate(child1, child2, eta=0.01)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                _ga.TB.mutate(mutant, eta=0.01)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(_ga.TB.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        _pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        _fits = [ind.fitness.values[0] for ind in _pop]

        length = len(_pop)
        mean = sum(_fits) / length
        sum2 = sum(x*x for x in _fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(_fits))
        print("  Max %s" % max(_fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        best_ind   = tools.selBest(_pop, 1)[0]
        best_value.append(best_ind.fitness.values[0])
        if _iter >= iter_min:
            eps = abs(best_ind.fitness.values[0] - best_value[0])
            del best_value[0]
        print(eps)
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values[0]))
    return np.array(best_ind), best_ind.fitness.values[0]

if __name__ == '__main__':
    x1_plot = np.linspace(-2, 2, 100)
    x2_plot = np.linspace(-2, 2, 100)
    x1_plot2, x2_plot2 = np.meshgrid(x1_plot, x2_plot)
        
    xlimits  = [[-2, 2], [-2, 2]]
    xlimits  = np.array(xlimits)
    sampling = LHS(xlimits=xlimits)
    # X        = sampling(2*len(xlimits))
    X        = [[1.5, -0.5], [0.5, -1.5], [-1.5, 1.5], [-0.5, 0.5]]
    X        = np.array(X)
    Y        = rosenbrock(X)
    _iter    = 0
    best_val = []
    output_ei  = open('output_ei.dat' , 'w')
    output_val = open('output_val.dat', 'w')
    while _iter < 40:
        _iter += 1
        print(X, Y)
        sm = krg(X, Y)
        best_ei,  ei_max  = main_ga(sm.EI, np.array(xlimits), 200, 100, 10000, addi=list(best_val))
        # best_val, fun_min = main_ga(sm.sm_ga, np.array(xlimits), 500, 100, 10000, weights=(-1,), addi=list(best_ei))
        best_val = np.where(Y == min(Y))
        best_val = X[best_val[0][0], :]
        
        # Using Adam
        def ei_one(X):
            a = sm.EI(X)
            return -a[0][0]

        def rosen_one(X):
            a = rosenbrock(X)
            return a[0]
            
        # Sub opt best ei
        with plotSurface(ei_one, xlimits, 100) as adam_plot:
            path = Path('./figure/ei_%i'%_iter)
            Path.mkdir(path, exist_ok=True)
            fig_cont   = adam_plot.create_fig('fcont')
            fig_3d     = adam_plot.create_fig('fplot3d')
            ax_contour = adam_plot.create_ax('fcont','cont', 111)
            ax_3d      = adam_plot.create_ax('fplot3d', 'plot3d', 111, projection='3d')
            cont = adam_plot.contourf('cont')
            plot3d = adam_plot.surface3d('plot3d')
            adam_plot.colorbar('fcont', cont)
            savename = path.joinpath('EI.png')
            adam_plot.save('fcont', savename)
            savename = path.joinpath('EI_sur.png')
            adam_plot.save('fplot3d', savename)
            adam_plot_x = []
            adam_plot_y = []
            adam_plot_z = []
            adam_plot_x2 = []
            adam_plot_y2 = []
            adam_plot_z2 = []
            plot_iter   = 0
            adam_opt  = Adam(ei_one, xlimits, lr=0.001)
            adam_opt2 = Adam(ei_one, xlimits, lr=0.001)
            for iter in range(3000):
                best_ei  = adam_opt.update(best_ei)
                ei_max   = -ei_one(best_ei)
                best_val = adam_opt2.update(best_val)
                val_max  = -ei_one(best_val)
                adam_plot_x.append(best_ei[0])
                adam_plot_y.append(best_ei[1])
                adam_plot_z.append(-ei_max)
                adam_plot_x2.append(best_val[0])
                adam_plot_y2.append(best_val[1])
                adam_plot_z2.append(-val_max)
                print(iter, best_ei,  ei_max)
                print(iter, best_val, val_max)
                if np.mod(iter, 100) == 0:
                    plot_iter += 1
                    adam_plot.scatter('cont',best_ei[0], best_ei[1])
                    adam_plot.line('cont',adam_plot_x, adam_plot_y)
                    adam_plot.scatter('cont',best_val[0], best_val[1], color='red')
                    adam_plot.line('cont',adam_plot_x2, adam_plot_y2, color='red')
                    # 3D-plot
                    adam_plot.line3d('plot3d', adam_plot_x, adam_plot_y, adam_plot_z)
                    adam_plot.scatter('plot3d',best_ei[0], best_ei[1], -best_ei, color='red')
                    adam_plot.line3d('plot3d', adam_plot_x2, adam_plot_y2, adam_plot_z2)
                    adam_plot.scatter('plot3d',best_val[0], best_val[1], -best_val, color='red')
                    savename = path.joinpath('EI_%i.png'%plot_iter)
                    savename = path.joinpath('EI_sur_%i.png'%plot_iter)
                    adam_plot.save('fcont', savename)
                    adam_plot.save('fplot3d', savename)
                    adam_plot_x = []
                    adam_plot_y = []
                    adam_plot_z = []
                    adam_plot_x2 = []
                    adam_plot_y2 = []
                    adam_plot_z2 = []
            print('----------------------------')
        
        # Sub opt best val
        # adam_opt2 = Adam(ei_one, xlimits, lr=0.01)
        # for iter in range(5000):
        #     best_val = adam_opt2.update(best_val)
        #     val_max  = -ei_one(best_val)
        #     print(best_val, val_max)
        # print('----------------------------')

        if val_max > ei_max:
            best_ei = best_val 

        # p_bfgs = fmin_l_bfgs_b(ei_one, best_ei, fprime=None, args=(), approx_grad=1, bounds=xlimits, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=- 1, maxfun=15000, maxiter=15000, disp=None, callback=None, maxls=20)
        # print(p_bfgs)
        # # Add point
        # best_ei_bfgs = np.atleast_2d(p_bfgs[0])
        # ei_max_bfgs  = np.array(p_bfgs[1])
        # print(best_ei_bfgs, ei_max_bfgs)
        # print('-----------------------')

        X = np.append(X, np.atleast_2d(best_ei), axis=0)
        Y = np.append(Y, rosenbrock(best_ei))
        # X = np.append(X, np.atleast_2d(best_val), axis=0)
        # Y = np.append(Y, rosenbrock(best_val))


        # # Plot the result
        # plot_ei  = np.zeros([100,100])
        # plot_val = np.zeros([100,100])
        # rosen    = np.zeros([100,100])
        # for i in range(100):
        #     for j in range(100):
        #         plot_ei [i, j] = sm.EI(np.atleast_2d([x1_plot[i], x1_plot[j]]))
        #         plot_val[i, j] = sm.sm_ga(np.atleast_2d([x1_plot[i], x1_plot[j]]))
        #         rosen[i, j]    = rosenbrock(np.atleast_2d([x1_plot[i], x1_plot[j]]))

        # for w in best_ei:
        #     output_ei.write('{:14.6f}'.format(w))
        # output_ei.write('{:14.6f}'.format(ei_max[0]))
        # output_ei.write('\n')

        # for w in best_val:
        #     output_val.write('{:14.6f}'.format(w))
        # output_val.write('{:14.6f}'.format(fun_min[0]))
        # output_val.write('\n')

        # fig = plt.figure(figsize=(14,9))
        # ax  = plt.axes(projection='3d')
        # ax.scatter(best_ei[0],best_ei[1],ei_max,color='black')
        # ax.plot_surface(x1_plot2, x2_plot2, plot_ei, cmap=cm.coolwarm, linewidth=0.2)
        # plt.savefig("GWO_ei_sm_{}.png".format(_iter)) 
        # plt.close()


        # fig2 = plt.figure(figsize =(14,9))
        # ax2 = plt.axes(projection='3d')
        # ax2.scatter(best_val[0],best_val[1],fun_min,color='black')
        # ax2.plot_surface(x1_plot2,x2_plot2,rosen,linewidth = 0.2,antialiased = True, color='grey') 
        # ax2.plot_surface(x1_plot2,x2_plot2,plot_val,linewidth = 0.2,antialiased = True,cmap='rainbow')
        # plt.savefig("GWO_sm_{}.png".format(_iter)) 

    output_ei.close()
    output_val.close()
