# 说明
sm_optimized 用于代理模型优化。

## 安装
- sm_optimized 目前仅可以从github获得。
```
git clone https://github.com/xilingchu/sm_optimized
```

- 使用前应将本模块上级目录放入python搜索路径中。

- sm_optimized 基于已有开源模块实现，包括但不限于smt、deap和scikit-opt。使用前应安装引用的模块，如：
```
pip install scikit-opt 
```
## 优化模块
- adam：adam算法基本模块
- ga：遗传算法实现
- PSO：带线性衰减惯性权重的粒子群算法实现

## 代理模型
- krg:带采集函数的克里格方法实现

## 测试函数
- funtool：
    - rosenbrock.py
    - branin.py
    - threeHumpCamel.py

## 工具函数
- utils：
    - plotlib:用于绘制测试图形的工具集
    - dataSample:提供二维情况下固定的LHS测试样本
    - loadCache:用于读取计算中保存的数据
