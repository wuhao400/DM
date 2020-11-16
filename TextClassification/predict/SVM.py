import datetime
import pickle
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
import pandas as pd
import os


def metrics_result(actual, predict):
    print('精度:{0:.3f}'.format(metrics.precision_score(actual, predict, average='weighted')))
    print('召回:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))
    print(classification_report(actual, predict))


def get_list(path):
    return os.listdir(path)


def linsvc_train(trainset, testset):
    '''

    :param trainset: 训练集经tfidf降维后的文件
    :param testset: 测试集经tfidf降维后的文件
    :return:

    https://zhuanlan.zhihu.com/p/57162092
    LinearSVC
    基于liblinear库实现
    有多种惩罚参数和损失函数可供选择
    训练集实例数量大（大于1万）时也可以很好地进行归一化
    既支持稠密输入矩阵也支持稀疏输入矩阵
    多分类问题采用one-vs-rest方法实现

    SVC
    基于libsvm库实现
    训练时间复杂度为 [公式]
    训练集实例数量大（大于1万）时很难进行归一化
    多分类问题采用one-vs-rest方法实现
    '''
    X_train = trainset.label
    y_train = trainset.tfidf_weight_matrics
    clf = LinearSVC(C=1, tol=1e-5)
    begintime_train = datetime.datetime.now()
    clf.fit(y_train, X_train)
    endtime_train = datetime.datetime.now()
    print("训练完毕，训练时长为：" + str((endtime_train - begintime_train).seconds)+ "秒")

    begintime_test = datetime.datetime.now()
    predicted = clf.predict(testset.tfidf_weight_matrics)
    metrics_result(testset.label, predicted)
    endtime_test = datetime.datetime.now()
    print("预测完毕，预测时长为：" + str((endtime_test - begintime_test).seconds)+ "秒")
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    list = get_list(r'D:\coding\newsdata_seg')
    print("混淆矩阵为：")
    print(pd.DataFrame(confusion_matrix(testset.label, predicted),
                       columns=list,
                       index=list))


if __name__ == '__main__':
    path = r'D:\coding\social'
    trainset_path = r"D:\coding\social\SVM1\train_tfidf_space.dat"
    testset_path = r"D:\coding\social\SVM1\test_tfidf_space.dat"
    with open(trainset_path, 'rb') as file_obj:
        trainset = pickle.load(file_obj)
    with open(testset_path, 'rb') as file_obj:
        testset = pickle.load(file_obj)
    linsvc_train(trainset, testset)
