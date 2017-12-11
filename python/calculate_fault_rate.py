# author: Luoxin
# -*- coding: utf-8 -*

from __future__ import division
from fileProcessing import *
import numpy as np


def calculate_fault_rate(index_class, X):

    # 计算每个DIMM中被分类为“class2（异常）”的样本数量
    dimm = np.zeros(4, dtype=int)
    for i in index_class:
        if (X[i][0] == 0 | X[i][0] == 1):
            dimm[0] += 1
        if (X[i][0] == 2 | X[i][0] == 3):
            dimm[1] += 1
        if (X[i][0] == 4 | X[i][0] == 5):
            dimm[2] += 1
        if (X[i][0] == 6 | X[i][0] == 7):
            dimm[3] += 1

    # 每执行一次算法得到一组故障率【dimm1_rate, dimm2_rate, ..., dimmn_rate】
    n_samples = X.shape[0]
    dimm_fault_rate = np.zeros(4, dtype=float)
    for i in range(len(dimm)):
        # 保留4位小数点
        r = round(dimm[i] / n_samples, 4)
        dimm_fault_rate[i] = r

    # 将所有DIMM的故障率追加写入以下文件中
    fp = open('fault_rate.txt', 'a')
    for r in dimm_fault_rate:
        msg = str(r) + '\t'
        fp.write(msg)
    fp.write('\n')
    fp.close()
