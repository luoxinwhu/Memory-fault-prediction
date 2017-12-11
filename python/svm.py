# author: Luoxin
# -*- coding: utf-8 -*


from findErrObj import *
from plotSVM import *
from sklearn.preprocessing import label_binarize

def svm_main(opt, C, gamma, X_train, y_train, X_test, y_test):
    '''
    SVM算法主函数
    :param opt: 核函数选择参数
    :param C:
    :param gamma:
    :param X_train: 属性训练集（未归一化）
    :param y_train: 标签训练集
    :param X_test:  属性测试集（未归一化）
    :param y_test:  标签测试集
    :return:
    '''
    random_state = np.random.RandomState(0)
    KERNEL = ['linear', 'rbf', 'poly', 'sigmoid']

    if opt==1:
        clf = svm.SVC(C=C,
           kernel='linear',
           probability=True,
           random_state=random_state,
           decision_function_shape='ovo')
    elif opt==2:
        clf = svm.SVC(
            C=C,
            gamma=gamma,
            kernel='rbf',
            probability=True,
            random_state=random_state,
            decision_function_shape='ovo')
    elif opt==3:
        clf = svm.SVC(
            C=C,
            gamma=gamma,
            coef0=0.1,
            degree=3,
            kernel='poly',
            probability=True,
            random_state=random_state,
            decision_function_shape='ovo')
    elif opt==4:
        clf = svm.SVC(
            C=C,
            gamma=gamma,
            coef0=0.1,
            kernel='sigmoid',
            probability=True,
            random_state=random_state,
            decision_function_shape='ovo')
    else:
        print "#error param: the legal opt is 1 or 2 or 3 or 4!"
        exit(-1)

    # 归一化属性集



    # fit()函数将训练数据集和训练标签集应用在定义好的分类器中，产生特定的预测模型
    clf.fit(X_train, y_train)

    # 预测模型应用decission_function()决策函数进行数据类别的预测
    # y_score - 是测试数据集通过模型预测出的标签在class0 和class1两种类别中的评分（概率值）
    y_score = clf.decision_function(X_test)

    # y_pred - 预测标签集
    y_pred = clf.predict(X_test)

    # look for data which is belonged to class_x in X_test
    # class2 - 异常
    # class1 - 次优
    # class0 - 优
    index_class0, index_class1, index_class2 = find_class_index(y_pred)

    # Compute ROC curve and ROC area for specific kernel model
    kernel = KERNEL[opt]
    y_test = label_binarize(y_test, classes=[0, 1, 2])
    plot_roc(y_test, y_score, 3, kernel=kernel)


    return index_class0, index_class1, index_class2