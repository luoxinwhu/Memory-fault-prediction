# author: Luoxin
# -*- coding: utf-8 -*

'''
Function:
    save_xls(label, data, filename)     写入label和data数组到指定的xls文件中，label固定存放在第0列
    load_xls(filename)                  加载指定的xls文件，读取出的数据存放在label和data这两个数组中
'''



import xlwt
import xlrd
import re
import numpy as np

def save_xls(label, data, filename):
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)

    for i, col in enumerate(label):
        sheet.write(i, 0, col)          # col[0] restore label Vector(m*1)

    for i, row in enumerate(data):
        for j, col in enumerate(row):
            sheet.write(i, 1+j, col)    # col[1~n] restore data Vector(m*n)

    workbook.save(filename)



def load_xls(filename):
    matrix = xlrd.open_workbook(filename)

    try:
        sheet = matrix.sheet_by_name('Sheet 1') # m*n Matrix
    except:
        print "No sheet in %s named Sheet1" % filename

    label = []
    for i in range(sheet.nrows):
        value = int(sheet.cell(i, 0).value)
        label.append(value)

    data = [[] for i in range(sheet.nrows)]
    for i in range(sheet.nrows):
        for j in range(1, sheet.ncols):
            data[i].append(sheet.cell_value(i,j))

    return label, data



def delIndexSymb(filename):
    '''
    删除属性单元格数值前的标号：  “1：-0.776”--> "-0.776"，并切分为label和properties两个数组
    '''
    f = open(filename)  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    N_FEATURES = 18
    #labelArr = []  # label
    propArr = []  # properties
    while line:
        # first = int(line[0:2])
        # labelArr.append(first)
        kvreg = re.compile("([0-9]+:[-.+0-9]+)+")
        res = kvreg.findall(line)
        vec = [0] * N_FEATURES
        for num in res:
            ans = num.split(':')
            vec[int(ans[0]) - 1] = int(ans[1])
        propArr.append(vec)
        line = f.readline()
    f.close()

    # assert len(labelArr) == len(propArr)
    propArr = np.asarray(propArr)
    return propArr



def appendTxt(filename, new_filename, input_array):
    '''
    按行读取filename文件内容，在每行内容前面添加input-array，新的内容存放在new_filename里面
    :param filename: 原文件
    :param new_filename: 新文件
    :param input_array: 需要添加的内容
    :return:
    '''
    input_array.tolist()
    dataArr = []
    # 读取一个旧文件内容
    fobj = open(filename)
    line = fobj.readline() # 行内容
    i=0
    while line:
        label = str(input_array[i])
        data = label+ '\t' +line
        dataArr.append(data)
        i += 1
        line = fobj.readline()
    fobj.close()

    # 创建并打开一个新文件
    fobj2 = open(new_filename, 'w')
    for data in dataArr:
        fobj2.write(data)
    fobj2.close()


