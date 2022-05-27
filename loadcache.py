# encoding=utf-8
import matplotlib as mpl
from sm_optimized.functool.loadCache import loadcache1
import matplotlib.pyplot as plt
import numpy as np

# 用于读取贝叶斯迭代过程数据的示例

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# cachelist = [
#     'ALL', 'EIonly', 'PIonly', 'Fonly', 'EI-F', 'PI-F', 'PI-EI', 'LCB'
# ] # 使用测试中定义的标识符
cachelist = [
    'psotest'
]

fig = plt.figure(figsize=(14, 9))
for Dir_flag in cachelist:
    BEST_Y, REAL_BEST_Y, ITER = loadcache1('./cache-' + Dir_flag + '/')
    plt.plot(ITER,
             REAL_BEST_Y - 0.397887,
             marker='o',
             mfc='w',
             label=u'REAL_BEST_Y ' + Dir_flag)
plt.legend()  # 让图例生效
plt.ylim(0, 3)  # 限定纵轴的范围
plt.xticks(range(np.max(ITER)), range(np.max(ITER)), rotation=45)
plt.margins(0)
plt.xlabel(u"")  # X轴标签
plt.ylabel("REAL_BEST_Y")  # Y轴标签
plt.title("REAL_BEST_Y")  # 标题
plt.show()
plt.close(fig)

fig = plt.figure(figsize=(14, 9))
for Dir_flag in cachelist:
    BEST_Y, REAL_BEST_Y, ITER = loadcache1('./cache-' + Dir_flag + '/')
    plt.plot(ITER,
             BEST_Y,
             marker='*',
             ms=10,
             label=u'PREDICTED BEST_Y ' + Dir_flag)

plt.legend()  # 让图例生效
# plt.ylim(-5, 5)  # 限定纵轴的范围
plt.xticks(range(np.max(ITER)), range(np.max(ITER)), rotation=45)
plt.margins(0)
plt.xlabel(u"")  # X轴标签
plt.ylabel("BEST_Y")  # Y轴标签
plt.title("PREDICTED_BEST_Y")  # 标题
plt.show()
plt.close(fig)
