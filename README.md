# DM
数据挖掘与数据仓库
使用语料库为样本，用Python实现文本分类
分类器包含Bayes和SVM两种
TextClassification为完整的项目文件
文本分类执行顺序：
1.爬取语料库
2.选择合适语料规模，形成训练集和测试集，并进行分词系统分词
3.使用Bunch结构化表示-构造词向量空间并持久化，可使用LDA建模
4.TF-IDF形成唯一向量空间，得到词典和权重矩阵tdm
5.分类器，训练，分类预测
6.分析和评价
参考:
1.https://blog.csdn.net/laobai1015/article/details/80415080
2.https://github.com/baixiaoyanvision/text_classify
