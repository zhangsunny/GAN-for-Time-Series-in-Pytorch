"""
使用分类器对时间序列进行分类，作为baseline
"""
import numpy as np
from utils.data_process import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy
from scipy.stats import wasserstein_distance


def load_data(name='ECG200', test_size=0.4):
    data, label = load_npz(name)
    x_train, x_test, y_train, y_test = \
        train_test_split(data, label, test_size=test_size)
    return (x_train, y_train), (x_test, y_test)


def classify_analysis(y, pred, labels=[0, 1], beta=1):
    """
    对二分类数据分类结果进行分析，默认使用labels[-1]作为positive class
    """
    conf_mat = confusion_matrix(y, pred, labels=labels)
    tn, fp, fn, tp = conf_mat.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f_beta = (beta**2 + 1)*precision*recall / ((beta**2)*precision + recall) \
        if ((beta**2)*precision + recall) != 0 else 0.0
    print('='*10+'Confusion Matirx'+'='*10)
    print('', conf_mat)
    print('=' * 11 + 'Classify Score' + '=' * 11)
    print('tn, fp, fn, tp:', tn, fp, fn, tp)
    print('Accuracy: %.4f' % accuracy)
    print('Precision: %.4f' % precision)
    print('Recall: %.4f' % recall)
    print('F-measure(beta=%s): %.4f' % (beta, f_beta))


def run_classify_analysis(model_name='SVC', model_args={},
                          labels=[-1, 1], name='ECG200', test_size=0.4):
    model_cls = {
        'SVC': SVC,
        'KNN': KNeighborsClassifier,
        'Tree': DecisionTreeClassifier,
    }
    if model_name not in model_cls.keys():
        raise VlaueError('Model name error!')
    (x_train, y_train), (x_test, y_test) = load_data(name, test_size)
    model = model_cls[model_name](**model_args)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    classify_analysis(y_test, pred, labels)


def KL_Wasserstein(data, pred):
    kl = np.mean(entropy(data, pred))
    wd = np.mean(wasserstein_distance(data, pred))
    return kl, wd

if __name__ == "__main__":
    run_classify_analysis('KNN', name='FordA')
