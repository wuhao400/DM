#!/usr/bin/env python

# -*- coding: UTF-8 -*-


import sys

#reload(sys)  # Python2.5 初始化后删除了 sys.setdefaultencoding 方法，我们需要重新载入
#sys.setdefaultencoding('utf-8')

import os  # python内置的包，用于进行文件目录操作，我们将会用到os.listdir函数

import pickle  # 导入cPickle包并且取一个别名pickle

from sklearn.utils import Bunch   # 一定是这样写的，下面注释的一行是错误示范
# from sklearn.datasets.base import Bunch

def readfile(path):
    '''
        读取文件
    '''

    with open(path, "r") as fp:  # with as句法前面的代码已经多次介绍过，今后不再注释

        content = fp.read()

    return content


def corpus2Bunch(wordbag_path, seg_path):

    catelist = os.listdir(seg_path)  # 获取seg_path下的所有子目录，也就是分类信息

    # 创建一个Bunch实例

    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])

    bunch.target_name.extend(catelist)

    ''''' 

    extend(addlist)是python list中的函数，意思是用新的list（addlist）去扩充 

    原来的list 

    '''

    # 获取每个目录下所有的文件

    for category in catelist:

        #class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径
        class_path = os.path.join(seg_path, category)

        file_list = os.listdir(class_path)  # 获取class_path下的所有文件
        print("Process: ", category)

        for file_path in file_list:  # 遍历类别目录下文件

            fullname = os.path.join(class_path, file_path)  # 拼出文件名全路径

            bunch.label.append(category)

            bunch.filenames.append(fullname)

            bunch.contents.append(readfile(fullname))  # 读取文件内容

        '''''append(element)是python list中的函数，意思是向原来的list中添加element，注意与extend()函数的区别'''

    # 将bunch存储到wordbag_path路径中

    with open(wordbag_path, "wb") as file_obj:

        pickle.dump(bunch, file_obj)

    print("构建文本对象结束！！！")

'''

wordbag_path = "D:\\Py_Learn\\textclassify_work\\results\\train_word_bag\\train_set.dat"  # Bunch存储路径

#seg_path = "D:/work/train/train_corpus_seg/"  # 分词后分类语料库路径
#seg_path = "D:\\Py_Learn\\textclassify_work\\results\\test_seg"
seg_path = "D:\\Py_Learn\\textclassify_work\\results\\train_corpus_seg"
corpus2Bunch(wordbag_path, seg_path)

'''


# 对测试集进行Bunch化操作：

wordbag_path = "D:\\Py_Learn\\textclassify_work\\results\\test_word_bag\\test_set.dat"  # Bunch存储路径
#wordbag_path = "D:/work/test/test_word_bag/test_set.dat"  # Bunch存储路径

#seg_path = "D:/work/test/test_corpus_seg/"  # 分词后分类语料库路径
seg_path = "D:\\Py_Learn\\textclassify_work\\results\\test_corpus_seg"
corpus2Bunch(wordbag_path, seg_path)
