# from krg_optimized.ga.ga import ga_op
from sm_optimized.functool.rosenbrock import rosenbrock
from sm_optimized.functool.branin import branin
# from sm_optimized.functool.dataSample import dataSample
from sm_optimized.krg.krgWithAllAC import krg
# from deap import tools
from smt.sampling_methods import LHS
import numpy as np
import matplotlib.pyplot as plt
from sko.PSO import PSO
# from krg_optimized.PSO.PSO import PSO  # 加线性权重
# from time import sleep
from sko.tools import set_run_mode
import pickle
import threading
import os, shutil


# 计算最优值
def PSO0RUN(pso0, _iter):
    set_run_mode(sm.EI, 'multithreading')
    pso0.run(200, 1e-6, 20)
    print('Max_EI_x is ', pso0.gbest_x, 'predicted Max_EI is', -pso0.gbest_y,
          'in iter', _iter, '\n')
    # run(max_iter=None, precision=None, N=20)
    '''
    precision: None or float
        If precision is None, it will run the number of max_iter steps
        If precision is a float, the loop will stop if continuous N difference between pbest less than precision
    N: int
    '''


def PSO1RUN(pso1, _iter):
    set_run_mode(sm.PI, 'multithreading')
    pso1.run(200, 1e-8, 20)
    print('Max_PI_x is ', pso1.gbest_x, 'predicted Max_PI is', pso1.gbest_y,
          'in iter', _iter, '\n')


def PSORUN(pso, _iter):
    set_run_mode(sm.sm_ga, 'multithreading')
    pso.run(max_iter=200, precision=1e-06)

    print('best_x is ', pso.gbest_x, 'predicted best_y is', pso.gbest_y,
          'in iter', _iter, '\n')


def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath, ignore_errors=True)
        os.mkdir(filepath)
    return filepath


if __name__ == '__main__':
    n_dim = 2  # 变量数（目前测试函数仅支持偶数）
    lb = []
    ub = []
    for i in range(0, n_dim):  # 定义变量上下界
        lb.append(-2)
        ub.append(2)
    xlimits = np.array([lb, ub])
    xlimits = np.array(xlimits.T)  # 定义变量范围（仅供LHS使用）
    mode = 0  # 1 从已有模型重启计算 / 0 从头运行
    _iter = 0
    ACfunctions = ['EI', 'F']  # 选定使用的采集函数，可选['EI','PI','LCB','F']
    Dir_flag = 'psotest'  # 定义本次测试的标识符
    cachedir = setDir('./cache-' + Dir_flag + '/')
    figdir = setDir('./fig-' + Dir_flag + '/')

    # 初始化绘图变量
    PLOT_flag = 1  # flag = 1 绘制图形，否则不输出图形
    BEST_Y = []
    REAL_BEST_Y = []
    ITER = []
    num_div = 100  # 二维图形网格密度
    x1_plot = np.linspace(lb[0], ub[0], num_div)
    x2_plot = np.linspace(lb[1], ub[1], num_div)
    x1_plot2, x2_plot2 = np.meshgrid(x1_plot, x2_plot)
    # rosen = np.zeros([num_div, num_div])
    plot_val = np.zeros([num_div, num_div])
    plot_ei = np.zeros([num_div, num_div])
    plot_pi = np.zeros([num_div, num_div])

    if mode == 0:
        sampling = LHS(xlimits=xlimits)  # 抽样
        X = sampling(2 * len(xlimits))
        Y = rosenbrock(X)
        # Y = branin(X)
        # X, Y = dataSample('branin')  # 使用二维固定样本

    while _iter == 0:  # 定义迭代计算次数
        if mode == 0:
            sm = krg(X, Y, _iter)
            with open("kriging.pkl", "wb") as f:  # 保存代理模型
                pickle.dump(sm, f)
        else:
            with open("kriging.pkl", "rb") as f:  # 读取代理模型
                sm = pickle.load(f)
                X = sm.X
                Y = sm.Y
                mode = 0
        ts = []  # 初始化线程表

        # 构建PSO优化问题
        pso = PSO(
            func=sm.sm_ga,  # 优化f(x)
            n_dim=len(lb),  # 问题维度
            pop=500,  # 粒子数量
            lb=lb,  # 变量下界
            ub=ub,  # 上界
            w=0.8,  # 惯性权重
            c1=1.2,  # 速度参数（全局偏向）
            c2=0.8,  # 速度参数（局部偏向）
            verbose=False)  # 是否输出优化过程
        ts.append(threading.Thread(target=PSORUN, args=[pso, _iter]))  # 添加线程

        if 'EI' in ACfunctions:
            pso0 = PSO(
                func=sm.EI,  # 优化EI（已取倒数）
                n_dim=len(lb),
                pop=500,
                lb=lb,
                ub=ub,
                w=0.6,
                c1=1.2,
                c2=0.8,
                verbose=False)
            ts.append(threading.Thread(target=PSO0RUN, args=[pso0, _iter]))

        if 'PI' in ACfunctions:
            pso1 = PSO(
                func=sm.PI,  # 优化PI
                n_dim=len(lb),
                pop=500,
                lb=lb,
                ub=ub,
                w=0.6,
                c1=1.2,
                c2=0.8,
                verbose=False)
            ts.append(threading.Thread(target=PSO1RUN, args=[pso1, _iter]))

        # 多线程计算
        for t in ts:
            t.start()
        for t in ts:
            t.join()

        # 存储值序列
        y = branin(np.array(np.float64(pso.gbest_x)))
        BEST_Y.append(pso.best_y)
        REAL_BEST_Y.append(y)
        ITER.append(_iter)
        print('REAL best_y is', y, '\n')

        # 加函数预测值点
        if 'F' in ACfunctions:
            X = np.append(X, np.atleast_2d(pso.gbest_x), axis=0)
            Y = np.append(Y, y)

        # 加EI预估点
        if 'EI' in ACfunctions:
            X = np.append(X, np.atleast_2d(pso0.gbest_x), axis=0)
            y = branin(np.array(pso0.gbest_x))
            Y = np.append(Y, y)

        # 加PI预估点
        if 'PI' in ACfunctions:
            X = np.append(X, np.atleast_2d(pso1.gbest_x), axis=0)
            y = branin(np.array(pso1.gbest_x))
            Y = np.append(Y, y)

        # %% 画图！
        if PLOT_flag == 1:
            for i in range(num_div):
                for j in range(num_div):
                    if 'EI' in ACfunctions:
                        plot_ei[i, j] = -sm.EI(
                            np.atleast_2d([x1_plot[j], x2_plot[i]]))
                    if 'PI' in ACfunctions:
                        plot_pi[i, j] = -sm.PI(
                            np.atleast_2d([x1_plot[j], x2_plot[i]]))
                    plot_val[i, j] = sm.sm_ga(
                        np.atleast_2d([x1_plot[j], x2_plot[i]]))
                    # rosen[i,
                    #   j] = branin(np.atleast_2d([x1_plot[j], x2_plot[i]]))
            # 画 EI分布
            if 'EI' in ACfunctions:
                fig = plt.figure(figsize=(14, 9))
                ctf = plt.contourf(x1_plot,
                                   x2_plot,
                                   plot_ei,
                                   100,
                                   cmap='rainbow')
                plt.colorbar(ctf)
                plt.plot(pso0.gbest_x[0], pso0.gbest_x[1], 'yo')
                plt.savefig(figdir + "MAX_EI_{}.png".format(_iter))
                plt.close(fig)

            # 画 PI分布
            if 'PI' in ACfunctions:
                fig = plt.figure(figsize=(14, 9))
                ctf = plt.contourf(x1_plot,
                                   x2_plot,
                                   plot_pi,
                                   100,
                                   cmap='rainbow')
                plt.colorbar(ctf)
                plt.plot(pso1.gbest_x[0], pso1.gbest_x[1], 'yo')
                plt.plot(pso.gbest_x[0], pso.gbest_x[1], 'bo')
                plt.savefig(figdir + "MAX_PI_{}.png".format(_iter))
                plt.close(fig)

            # 画代理模型值分布
            fig = plt.figure(figsize=(14, 9))
            ctf = plt.contourf(x1_plot, x2_plot, plot_val, 100, cmap='hot')
            plt.colorbar(ctf)
            plt.plot(pso.gbest_x[0], pso.gbest_x[1], 'bo')
            plt.savefig(figdir + "Predict_Y_{}.png".format(_iter))
            plt.close(fig)
            plt.close('all')

        # 保存本次计算数据
        np.savez(cachedir + 'data.npz',
                 REAL_BEST_Y=np.array(REAL_BEST_Y),
                 BEST_Y=np.array(BEST_Y),
                 ITER=np.array(ITER))

        _iter += len(ACfunctions)
