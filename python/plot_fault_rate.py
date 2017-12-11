# author: Luoxin
# -*- coding: utf-8 -*

'''
============================
 plot_fault_rate Function
============================
Funtion execution time = n * td, td is observation time interval of the classification algorithm,
and n is the number of execution of the classification algorithm

This function will plot fault rate - time curves of different DIMMs.
From this figure, we can judge the healthy-situation of the specific DIMM
'''
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np


fault_rate_arr = np.loadtxt('fault_rate.txt')
N_TIME, N_DIMM = fault_rate_arr.shape
fault_rate_arr = fault_rate_arr.reshape(N_DIMM, N_TIME)
time = range(0,N_TIME,1)

plt.figure()
for row in range(N_DIMM):
    label = 'DIMM'+str(row)
    plt.plot(time, fault_rate_arr[row]*100,'*-', label=label)

plt.title("Fault rate with time curves of different DIMMs", fontsize=12, fontweight = 'bold')
plt.ylabel("fault rate %")
plt.xlabel("time (t0 is original time, td is observation time interval)")
plt.legend(fontsize=8)
plt.grid()
plt.ylim(0, 100)    # 设置y轴的范围
x_ticks = []
for i in range(0,N_TIME):
    s = 't0+'+str(time[i])+'td'
    x_ticks.append(s)
plt.xticks(time, x_ticks, fontsize=10)  # 设置刻度字体大小
plt.show()

print "----------------\nend!"

