#!/usr/bin/env python

# -*- coding: UTF-8 -*-


import sys

import os
#分词库

import pynlpir

# 配置utf-8输出环境
#reload(sys)  弃用
import importlib
importlib.reload(sys)
#sys.setdefaultencoding('utf-8')


# 保存至文件

def savefile(savepath, content):
    with open(savepath, "w") as fp:
        fp.write(content)
       # print("wh:"+savepath)

# 读取文件

def readfile(path):
    with open(path, "r") as fp:
        content = fp.read()

    return content


def corpus_segment(corpus_path, seg_path):
    '''''

    corpus_path是未分词语料库路径

    seg_path是分词后语料库存储路径

    '''
    max_seg = 80000
    max_train_seg = 70000
    pynlpir.open()     #分词系统
    catelist = os.listdir(corpus_path)  # 获取corpus_path下的所有子目录

    ''''' 
     py_data01/留学/
    其中子目录的名字就是类别名，例如： 
    '''

    # 获取每个目录（类别）下所有的文件
    finish = ['产经', '法治', '房产', '教育', '金融', '军事', '能源', '台湾', '文化', '证券']
    for category in catelist:

        ''''' 

        这里category就是类别,如军事 

        '''
        if category not in finish:

           i = 0
           flag = 0
           #class_path = corpus_path + category + "/"  # 拼出分类子目录的路径如：train_corpus/art/
           class_path = os.path.join(corpus_path, category)
           #seg_dir = seg_path + category + "/"  # 拼出分词后存贮的对应目录路径如：train_corpus_seg/art/
           seg_dir = os.path.join(seg_path, category)
           seg_test_dir = os.path.join("D:\\Py_Learn\\textclassify_work\\results\\test_corpus_seg", category)
           #print("wh-1",seg_dir)
           if not os.path.exists(seg_dir):  # 是否存在分词目录，如果没有则创建该目录

              os.makedirs(seg_dir)
           if not os.path.exists(seg_test_dir):  # 是否存在分词目录，如果没有则创建该目录

              os.makedirs(seg_test_dir)

           years = os.listdir(class_path)  # 获取未分词语料库中某一类别中的所有文本


           for year in years:  # 遍历类别目录下的所有文件

              if flag == 1:
                  break
              yearname = os.path.join(class_path, year)  # 拼出文件名年份路径如：train_corpus/art/21.txt
              months = os.listdir(yearname)

              for month in months:

                if flag == 1:
                    break
                print("cleaning:"+category+"month"+month)
                month_path = os.path.join(yearname, month)
                raw_path = os.listdir(month_path)
                for document in raw_path:
                    i += 1
                    if i > max_seg:
                        flag = 1
                        break
                    fullname = os.path.join(month_path, document)
                    content = readfile(fullname)  # 读取文件内容
                    '''''此时，content里面存贮的是原文本的所有字符，例如多余的空格、空行、回车等等， 

                          接下来，我们需要把这些无关痛痒的字符统统去掉，变成只有标点符号做间隔的紧凑的文本内容 

                     '''
                    #content = content.replace('\n', '')  # 删除换行
                    #content = content.replace(' ', '')  # 删除空行、多余的空格
                    try:
                        con_segx = pynlpir.segment(content, pos_english=True)
                    except UnicodeDecodeError:
                        print(category+" "+document+" UnicodeDecodeError_wh")
                    content_seg=[element[0] for element in con_segx if element[1] == 'noun']
                    #content_seg = jieba.cut(content)  # 为文件内容分词
                    if i <= max_train_seg:
                        savefile(seg_dir + "\\" + document, " ".join(content_seg))  # 将处理后的文件保存到分词后语料目录
                    else:
                        savefile(seg_test_dir + "\\" + document, " ".join(content_seg))
    pynlpir.close()

    print("中文语料分词结束！！！")


''''' 

if __name__=="__main__":

简单来说如果其他python文件调用这个文件的函数，或者把这个文件作为模块 

导入到你的工程中时，那么下面的代码将不会被执行，而如果单独在命令行中 

运行这个文件，或者在IDE（如pycharm）中运行这个文件时候，下面的代码才会运行。 

即，这部分代码相当于一个功能测试。 



'''

if __name__ == "__main__":
    # 对训练集进行分词
    # corpus_path = "D:/work/train/train/"  # 未分词分类语料库路径
    corpus_path = "D:\\Py_Learn\\textclassify_work\\py_data02"
    # seg_path = "D:/work/train/train/train_corpus_seg/"  # 分词后分类语料库路径
    seg_path = "D:\\Py_Learn\\textclassify_work\\results\\train_corpus_seg"
    corpus_segment(corpus_path, seg_path)
'''
    # 对测试集进行分词

    corpus_path = "D:/work/test/test/"  # 未分词分类语料库路径
    #  "D:\\Py_Learn\\textclassify_work\\py_data03"
    seg_path = "D:/work/test/test/test_corpus_seg/"  # 分词后分类语料库路径
    # "D:\\Py_Learn\\textclassify_work\\results\\test_corpus_seg"
    corpus_segment(corpus_path, seg_path)
'''