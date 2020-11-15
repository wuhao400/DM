# !/usr/bin/env python

# -*- coding: UTF-8 -*-


import pickle
import pandas as pd
import numpy

from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# 读取bunch对象

def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)

    return bunch


# 导入训练集

#trainpath = "D:/work/train/train_word_bag/tfdifspace.dat"
trainpath = "D:\\Py_Learn\\textclassify_work\\results\\train_word_bag\\train_tfdifspace.dat"

train_set = _readbunchobj(trainpath)

# 导入测试集

#testpath = "D:/work/test/test_word_bag/testspace.dat"
#testpath = "D:\\Py_Learn\\textclassify_work\\results\\train_word_bag\\train_tfdifspace.dat"
testpath = "D:\\Py_Learn\\textclassify_work\\results\\test_word_bag\\test_tfdifspace.dat"
test_set = _readbunchobj(testpath)

# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高

clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)

# 预测分类结果

predicted = clf.predict(test_set.tdm)

# 打印预测错误文档
error_predict_text = "D:\\Py_Learn\\textclassify_work\\results\\predicts\\Bayes\\Re_02.log"
fz = open(error_predict_text, 'w')
for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):

    if flabel != expct_cate:

        fz.write(file_name+": 实际类别:"+flabel+" -->预测类别:"+expct_cate)
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)
fz.close()


# 计算分类精度：

save_metric_result_path = "D:\\Py_Learn\\textclassify_work\\results\\predicts\\Bayes\\Re_01.log"

def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))

    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))

    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))

    print("classification_report: "+classification_report(actual, predict))

    fs = open(save_metric_result_path, 'w')  # 存储文本
    fs.write('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    fs.write('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    fs.write('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
    fs.write(classification_report(actual, predict)+"\n\n")



metrics_result(test_set.label, predicted)
print("预测完毕!!!")