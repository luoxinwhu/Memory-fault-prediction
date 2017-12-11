# author: Luoxin
# -*- coding: utf-8 -*

"""
=======================
  SVM algorithm entry
=======================

This is a two-classes classification problem.The input data sets
come from a specific program that is responsible for collecting
real values of some model specific registers(MSRs).
"""
print(__doc__)


from svm import *
from svr import *
from dataProcessing import *
from fileProcessing import *
from grid import *
from plotSVM import *
from findErrObj import *
from calculate_fault_rate import *

'''
--------------------
svm classification 
--------------------
'''
# delete the mark number in the front of each property values.
X = delIndexSymb('a.txt')
# define label of original data sets
y = define_label(X)

# shuffle and split training and test sets
# X - 是数据集; y - 是标签集
print ">> Data and label sets are splitting"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0) #随机划分


# grid search to find the optimal parameters for model train
print ">> Grid Search Algorithm is running..."
appendTxt(filename='a.txt', new_filename='a2.txt', input_array=y)
grid_rate, grid_param = find_parameters('a2.txt', '-log2c 0,15,1 -log2g -15,-30,-1')
C = grid_param['c']
gamma = grid_param['g']

# class2 - 异常; class1 - 次优; class0 - 优
index_class0, index_class1, index_class2 = svm_main(3, C, gamma, X_train, y_train, X_test, y_test)
calculate_fault_rate(index_class1, X)



# regression
#svr(X, y)


"""
------------------------
    Figure/Result Area
------------------------
"""
#plot data sample with different classifier kernel
#plot_sample(X_test, y_test, C, gamma)

# print rankID, channelID with the error data record
print_rankID_channelID(index_class2, X_test)


# plot corrected error bar figure of eight different ranks
CE_total = get_CE_total(X_test)           # get correctable errors total number of the ranks
CE_min, CE_max = get_CE_min_max(X_test)   # get the minimum number of the ranks
plot_CE_bar(CE_total, CE_min, CE_max)

# plot corrected error number-time curve of eight different ranks
CE_number = get_CE_number(X_test, timeWidth=2)
plot_CE_time(CE_number, timeWidth=2)

print "\n-- SVM Algorithm is already done --"