# author: Luoxin
# -*- coding: utf-8 -*
'''
=======================================================================================
                                Function List
=======================================================================================
    splitArr(array)      提取[[a,b],[c],[d,e,f]]格式数组中的每个子数组的某一列，得到[a,c,d]
    np.ravel(array)      转换[[a,b],[c],[d,e,f]]格式为[a, b, c, d, e, f]
    label_unbinarize     将二值化标签转换为十进制原始标签
=======================================================================================
'''
import numpy as np
import math
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import sys


def splitArr(array):
    # 提取[[a1, b1],[a2,b2]...[an,bn]]中的每个元素的第一列和第二列
    # 得到[[a1, a2, a3,...,an],[b1, b2, ..., bn]]
    output = [[], []]
    for i in range(0, len(array)):
        output[0].append(array[i][0])
        output[1].append(array[i][1])
        i += 1

    return output

# np.ravel() can transfer [[],[],[]...] to [, , ,] structure

def label_unbinarize(labelArr, classes, pos_label=1, other_label=2):

    labels = []

    # 遍历每一组标签
    for elem in labelArr:
        # 遍历某个标签组的每个元素，查找值为pos_label的元素所在的位置，用此位置索引类别模板
        for index, val in enumerate(elem):
            if(val==pos_label):
                labels.append(classes[index])
                break
            if(index == len(elem)-1):
                labels.append(other_label)
                break

    return labels

# def label_binarize(labels, classes, neg_label=0, pos_label=1):
#
#     labelArr = []
#
#     for elem in labels:
#         tmp = [neg_label, neg_label]
#         for index,val in enumerate(classes):
#             if(elem==val):
#                 tmp[index] = pos_label
#         labelArr.append(tmp)
#
#     return labelArr

def define_label(X):
    n_samples, n_features = X.shape
    y = []

    OVF_GOOD = 4
    OVF_PERFECT= 0
    CNT_GOOD = round(99*8*0.6)
    CNT_PERFECT = round(99*8*0.4)

    for i in range(0, n_samples):
        ov = 0
        cnt = 0
        j=2
        while j<=16:
            cnt += X[i][j]
            # overflow属性索引[3,5,7,9,11,13,15,17]
            ov += X[i][j+1]
            j += 2

        # judge
        if(ov<=OVF_PERFECT):
            if(cnt<=CNT_PERFECT):
                label = 0
            else:
                label = 1
        elif(ov<=OVF_GOOD):
            if(cnt<=CNT_GOOD):
                label = 1
            else:
                label = 2
        else:
            label = 2
        y.append(label)

    y = np.asarray(y)
    return y