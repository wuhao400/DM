# !/usr/bin/env python

# -*- coding: UTF-8 -*-

import pickle

from sklearn.utils import Bunch

from sklearn.feature_extraction.text import TfidfVectorizer


def _readfile(path):
    with open(path, "rb") as fp:
        content = fp.read()

    return content


def _readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)

    return bunch


def _writebunchobj(path, bunchobj):
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

'''
    #函数说明:生成N篇文档的TF-IDF向量空间
    :param bunch_path:输入的词袋文件路径
    :param space_path:输出的TF-IDF特征空间路径
    :param train_tfidf_path:训练集的特征空间，词典供测试集使用
'''

def vector_space(bunch_path, space_path, train_tfidf_path=None):
    #stpwrdlst = _readfile(stopword_path).splitlines()

    bunch = _readbunchobj(bunch_path)
    #param tdm: 词频矩阵

    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})

    if train_tfidf_path is not None:         #测试集

        trainbunch = _readbunchobj(train_tfidf_path)

        tfidfspace.vocabulary = trainbunch.vocabulary   #使用训练集的字典

        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.1, min_df=0.001,
                                     vocabulary=trainbunch.vocabulary)

        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)



    else:                        #训练集

        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.1, min_df=0.001)

        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)

        tfidfspace.vocabulary = vectorizer.vocabulary_


    _writebunchobj(space_path, tfidfspace)

    print("tf-idf词向量空间实例创建成功！！！")


if __name__ == '__main__':
    #stopword_path = "D:/work/train/train_word_bag/hlt_stop_words.txt"
    #bunch_path = "D:/work/train/train_word_bag/train_set.dat"
    bunch_path = "D:\\Py_Learn\\textclassify_work\\results\\train_word_bag\\train_set.dat"

    #space_path = "D:/work/train/train_word_bag/tfdifspace.dat"
    space_path = "D:\\Py_Learn\\textclassify_work\\results\\train_word_bag\\train_tfdifspace.dat"

    vector_space(bunch_path, space_path)

    #bunch_path = "D:/work/test/test_word_bag/test_set.dat"
    bunch_path = "D:\\Py_Learn\\textclassify_work\\results\\test_word_bag\\test_set.dat"
    #space_path = "D:/work/test/test_word_bag/testspace.dat"
    space_path = "D:\\Py_Learn\\textclassify_work\\results\\test_word_bag\\test_tfdifspace.dat"
    #train_tfidf_path = "D:/work/train/train_word_bag/tfdifspace.dat"
    train_tfdif_path = "D:\\Py_Learn\\textclassify_work\\results\\train_word_bag\\train_tfdifspace.dat"
    vector_space(bunch_path, space_path, train_tfdif_path)