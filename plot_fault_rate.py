# author: Luoxin
# -*- coding: utf-8 -*



def plot_fault_rate(fault_rate):
    t = range(0, 10, 1)

    plt.figure()
    plt.plot(fault_rate, t)