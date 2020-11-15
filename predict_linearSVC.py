
import os
import pickle
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


def _readbunchobj(path):
    with open(path, 'rb') as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

save_metric_result_path = "D:\\Py_Learn\\textclassify_work\\results\\predicts\\Svm\\Re_LinearSVC01.log"
trainpath = "D:\\Py_Learn\\textclassify_work\\results\\train_word_bag\\train_tfdifspace.dat"
train_set = _readbunchobj(trainpath)
#testpath = "D:\\Py_Learn\\textclassify_work\\results\\train_word_bag\\train_tfdifspace.dat"
testpath = "D:\\Py_Learn\\textclassify_work\\results\\test_word_bag\\test_tfdifspace.dat"
test_set = _readbunchobj(testpath)
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
    fs.close()


def linear_svc_train(trainset, testset):
    '''
     训练过程
    :param trainset: tf_idf训练集
    :param testset:  tf_idf测试集
    :return:
    '''
    print("正在使用L_SVC训练...")
    clf =LinearSVC(C=1, tol=1e-5).fit(trainset.tdm, trainset.label)
    print("预测中...")
    predicted = clf.predict(testset.tdm)
    metrics_result(testset.label, predicted)

linear_svc_train(train_set,test_set)








