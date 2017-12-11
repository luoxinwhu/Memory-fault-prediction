# author: Luoxin
# -*- coding: utf-8 -*

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
from itertools import cycle
from scipy import interp
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def plot_roc(y_test, y_score, n_classes, kernel):

    print ">> plot_roc.py is called"

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    len = y_test.shape[0]
    for i in range(0,len):
        fpr[i], tpr[i], _ = roc_curve(y_test[:][i], y_score[:][i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot of a ROC curve for a specific class
    lw = 2
    # plt.figure()
    # plt.plot(fpr[0], tpr[0], color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()


    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve based on ' + kernel + ' kernel')
    plt.legend(loc="lower right")
    plt.savefig("../results/ROC-" + kernel)
    plt.close()


def plot_roc_crossval(X,y):
    print ">> plot_roc_crossval.py is called"

    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits = 10)
    classifier = svm.SVC(kernel='rbf', probability=True, random_state=random_state)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',lw=2, alpha=.8,
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves with cross validation')
    plt.legend(loc="lower right")
    #plt.show()
    plt.savefig('../results/ROC-cross validation')
    plt.close()


def plot_sample(X,y,C, gamma):

    print '>> plot_sample.py is called'

    def make_meshgrid(x, y, step=.01):
        """创建一个点的网格用以绘图

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        step: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                             np.arange(y_min, y_max, step))
        return xx, yy


    def plot_contours(ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf等位线, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out


    # Take the first two features. We could avoid this by using a two-dim dataset
    X = X[:, :2]

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='sigmoid', coef0=3.5,gamma=gamma ,C=C),
              svm.SVC(kernel='poly', degree=3, coef0=0, C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with sigmoid kernel(coef0=3.5)',
              'SVC with polynomial kernel(degree=3)')

    # Set-up 2x2 grid for plotting.
    sub1 = plt.subplot(321)
    sub2 = plt.subplot(322)
    sub3 = plt.subplot(323)
    sub4 = plt.subplot(324)
    sub5 = plt.subplot(325)
    sub = np.asarray([sub1,sub2,sub3,sub4,sub5])
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title, fontsize=11, fontweight='bold')

    plt.savefig('../results/sample classification')
    plt.close()


def plot_CE_bar(CE_total, CE_min, CE_max):
    '''
    绘制在某个时间段内八个rank内CE的总数、最大值、最小值的柱状图
    :param CE_total: CE总数
    :param CE_min: CE最小值
    :param CE_max: CE最大值
    :return: none
    '''
    plt.figure()
    n_groups = 8
    index = np.arange(n_groups)
    bar_width = 0.2
    distance = 1.5*bar_width
    opacity = 0.5
    # 绘图 pyplot.bar(left, height, width=0.8, bottom=None, hold=None, **kwargs)
    plt.bar(index, CE_total, bar_width, alpha=opacity, color='b', label='total')
    plt.bar(index+bar_width, CE_min, bar_width, color='r', label='min')
    plt.bar(index+bar_width*2, CE_max, bar_width, alpha=opacity, color='y', label='max')

    # 标注数值
    for a, b in zip(index, CE_total):
        plt.text(a, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    for a,b in zip(index, CE_min):
        plt.text(a+bar_width, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)
    for a,b in zip(index, CE_max):
        plt.text(a+bar_width*2, b, '%.0f' % b, ha='center', va='bottom', fontsize=8)

    plt.title('The corrected error number of eight ranks')
    plt.xlabel('Rank Groups')
    plt.ylabel('Corrected error number')
    # 设置每个分组的x轴上的名称
    plt.xticks(index+distance, ('Rank0', 'Rank1', 'Rank2', 'Rank3', 'Rank4', 'Rank5', 'Rank6', 'Rank7'))
    plt.legend(loc='upper right', ncol=3)
    plt.savefig('../results/CE_bar')
    plt.close()


def plot_CE_time(CE_number, timeWidth):
    '''

    :param CE_number: [8, n] , 每一行代表一个rank，列记录了某个时刻CE的数量
    :param timeWidth: 时间序列
    :return:
    '''
    RANK = ['rank0', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5', 'rank6', 'rank7']
    time = timeWidth*np.asarray(range(1,CE_number.shape[1]+1)) # 距离起始时间的时间累积长度
    x_ticks = []
    for i in range(0,time.shape[0]):
        s = 't0+'+str(time[i])+'ts'
        x_ticks.append(s)

    # 将8个rank的情况绘制在一张图中
    plt.figure(figsize=(12,8))
    for index,rows in enumerate(CE_number):
        plt.plot(time, rows[:], '*--', label=RANK[index])
    plt.title('CE number-time variation curve of different ranks', fontsize=14, fontweight='bold')
    plt.xlabel('Cumulative time\nt = t0 + x(i)*ts, t0 is the starting time and ts is the sampling time', fontsize=12)
    plt.ylabel('Corrected error number', fontsize=12)
    plt.legend(loc='best', ncol=8, fontsize=10)
    # 设置刻度字体大小
    plt.xticks(time,x_ticks, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid()
    plt.savefig('../results/CE_time')
    plt.close()


    # 分别绘制单个rank中CE的变化情况
    for index,rows in enumerate(CE_number):
        plt.figure()
        plt.plot(time, rows[:], '*--', label=RANK[index])
        plt.title('CE number-time variation curve of different ranks', fontsize=12, fontweight='bold')
        plt.xlabel('Cumulative time\nt = t0 + x(i)*ts, t0 is the starting time and ts is the sampling time', fontsize=10)
        plt.ylabel('Corrected error number', fontsize=10)
        plt.legend(loc='best',fontsize=8)
        # 设置刻度字体大小
        plt.xticks(time,x_ticks, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid()
        plt.savefig('../results/CE_time_'+RANK[index])
        plt.close()