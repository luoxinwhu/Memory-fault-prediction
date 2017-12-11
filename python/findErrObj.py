# author: Luoxin
# -*- coding: utf-8 -*

import collections
import numpy as np



def find_class_index(y_pred):
    """
    在测试集中寻找判别为class_x的数据
    :param y_pred: 预测标签集
    :return index_class
    """
    index_class0 = []
    index_class1 = []
    index_class2 = []

    for index, val in enumerate(y_pred):
        if(val == 2):
            index_class2.append(index)
        if(val == 1):
            index_class1.append(index)
        if(val == 0):
            index_class0.append(index)

    print "在测试样本中通过SVM算法预测归类于class_0的数据索引为:",index_class0
    print "在测试样本中通过SVM算法预测归类于class_1的数据索引为:",index_class1
    print "在测试样本中通过SVM算法预测归类于class_2的数据索引为:",index_class2
    return index_class0, index_class1, index_class2



#
# def find_sameClass_index(index1, index2, index3, index4, class_x):
#     index_sameClass = []
#     total_index = index1 + index2 + index3 + index4
#     total_index.sort()
#
#     # 获取total_errLabelPos数组中每个元素出现的次数
#     d = collections.Counter(total_index)
#     #print d
#
#     print "综合四种预测模型，得到属于class_"+str(class_x)+"的数据索引(默认从0开始)"
#     for k in d:
#         print "行数：",k,"\t重复次数：",d[k]
#         index_sameClass.append(k)
#
#     return index_sameClass


# def find_diffLabel(y_test, y_pred):
#     errLabelPos = []
#     y_test = y_test.tolist()
#     y_pred = y_pred.tolist()
#     for i in range(0,len(y_test)):
#         if(y_test[i]!=y_pred[i]):
#             errLabelPos.append(i)
#     return errLabelPos



# def find_errorDataRecords(errLabelPos):
#     #  根据错误预测标签索引在测试集中寻找标签分类为“0”的数据记录（正确预测类别应该为1）


def print_rankID_channelID(err_data_index, X_test):
    print "\nThe channelID and rankID of the Error data which are belonged to class2 are as follows:"
    for pos in err_data_index:
        print "\tchannelID: ", X_test[pos][1], "rankID: ", X_test[pos][0]
    print "-------------------------------------------"
    print "These DIMMs need to be changed: ", "DIMM1\n"



def get_CE_total(X_test):
    '''
    统计在相同时间段内，rank0~rank7中产生的corrected error总数
    :param X_test:
    :return: CE_total[1*8] 每一位对应一个rank中产生的CE总数
    '''
    CE_total = np.zeros(8)

    for row in X_test:
        j=0
        for i in [2,4,6,8,10,12,14,16]: # 选中属性矩阵中errcnt_x（八个rank的Corrected error数量）
            CE_total[j] += row[i] # 计算在每个rank中CE的总量
            j = j+1

    CE_total.tolist()
    print "\n在本次观察周期内rank0~rank7中发生的可纠正错误的数量分别为：\n", CE_total
    return CE_total


def get_CE_min_max(X_test):
    '''
    统计在相同时间段内，rank0~rank7中产生的corrected error最大值和最小值
    :param X_test:
    :return:
    '''
    CE_min = 200*np.ones(8) # 可纠正错误最少出现的次数，初始化200（单条记录中errcnt_x<200)
    CE_max = np.zeros(8)    # 可纠正错误最少出现的次数，初始化0（单条记录中errcnt_x>=0)

    for row in X_test:
        for i in [2,4,6,8,10,12,14,16]: # 选中属性矩阵中errcnt_x（八个rank的Corrected error数量）
            j = i/2-1
            if(row[i]<CE_min[j]):
                CE_min[j] = row[i]
            if(row[i]>CE_max[j]):
                CE_max[j] = row[i]

    print "\n在本次观察周期内rank0~rank7中发生的可纠正错误的最小次数分别为：\n", CE_min
    print "\n在本次观察周期内rank0~rank7中发生的可纠正错误的最多次数分别为：\n", CE_max
    return CE_min, CE_max


def get_CE_number(X_test, timeWidth):
    '''
    获取时间累积CE数量
    :param X_test: 测试集 【n_samples , n_features】
    :param timeWidth: 时间累积宽度
    :return: CE_number【8,  n_samples/timeWidth】，每一行表示rank_x，每一列表示在每个时间区间内的CE累积数量
    '''
    n_samples, n_features = X_test.shape
    CE_number = np.zeros((8, n_samples/timeWidth))

    X_test = X_test.reshape(n_features, n_samples)
    for row in [2,4,6,8,10,12,14,16]:
        col=0
        while col<n_samples-timeWidth:
            number = 0
            for j in range(0, timeWidth):
                number += X_test[row][col+j]
            CE_number[row/2-1][col/2]=number
            col += timeWidth

    return CE_number


# x = np.asarray([[1,1,100,1,102,11,109,11,111,12,124,12,1234,13,109,1,124,1],
#                [2,2,2234,2,254,2,255,2,275,2,277,2,299,2,288,2,266,2],
#                [3,4,344,4,267,5,457,4,657,7,746,8,636,6,547,2,4758,4],
#                [5,3,635,7,375,6,757,4,6678,8,7567,5,4567,3,654,7,490,3]])
#
# c = get_CE_number(x, 2)











