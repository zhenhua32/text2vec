# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from loguru import logger
import random
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch


def set_seed(seed):
    """
    Set seed for random number generators.
    """
    logger.info(f"Set seed for random, numpy and torch: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def l2_normalize(vecs):
    """
    L2标准化
    """
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


"""
Spearman相关系数是一种衡量两个变量之间单调关系强度的非参数统计方法，它使用变量的排序位次而不是原始数据来计算。伪代码实现如下：

```python
# 定义一个函数，计算一个列表的排序位次
def rank(x):
  # 创建一个空字典，存储元素和位次的对应关系
  rank_dict = {}
  # 对列表进行排序，得到一个新的列表
  sorted_x = sorted(x)
  # 遍历排序后的列表，给每个元素赋予一个位次，从1开始
  for i in range(len(sorted_x)):
    rank_dict[sorted_x[i]] = i + 1
  # 返回一个列表，包含原列表中每个元素的位次
  return [rank_dict[e] for e in x]

# 定义一个函数，计算两个列表之间的Spearman相关系数
def spearman_corr(x, y):
  # 检查两个列表长度是否相等，如果不等则抛出异常
  if len(x) != len(y):
    raise ValueError("The lengths of x and y must be equal.")
  # 计算两个列表的排序位次
  rx = rank(x)
  ry = rank(y)
  # 计算两个排序位次列表之间的皮尔逊相关系数
  return pearson_corr(rx, ry)

# 定义一个函数，计算两个列表之间的皮尔逊相关系数
def pearson_corr(x, y):
  # 检查两个列表长度是否相等，如果不等则抛出异常
  if len(x) != len(y):
    raise ValueError("The lengths of x and y must be equal.")
  # 计算两个列表的均值
  mean_x = sum(x) / len(x)
  mean_y = sum(y) / len(y)
  # 计算两个列表的协方差和标准差
  cov = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
  std_x = (sum([(x[i] - mean_x) ** 2 for i in range(len(x))]) / len(x)) ** 0.5
  std_y = (sum([(y[i] - mean_y) ** 2 for i in range(len(y))]) / len(y)) ** 0.5
  # 计算并返回皮尔逊相关系数
  return cov / (std_x * std_y)
```

源: 与必应的对话， 2023/4/1(1) 非参数检验 | Spearman 秩相关检验 - 知乎. https://zhuanlan.zhihu.com/p/513741141 访问时间 2023/4/1.
(2) 斯皮尔曼相关(Spearman correlation)系数概述及其计算例_斯皮尔曼秩相关系数_笨牛慢耕的博客-CSDN博客. https://blog.csdn.net/chenxy_bwave/article/details/121427036 访问时间 2023/4/1.
(3) Spearman 相关性分析法,以及python的完整代码应用_Freshman小白的博客-CSDN博客. https://blog.csdn.net/weixin_67016521/article/details/129863814 访问时间 2023/4/1.
"""


def compute_spearmanr(x, y):
    """
    Spearman相关系数
    """
    return spearmanr(x, y).correlation


def compute_pearsonr(x, y):
    """
    Pearson系数
    """
    return pearsonr(x, y)[0]
