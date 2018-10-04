# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:53:33 2017

@author: thinkwee
"""

from Tools.data_util import DataUtils
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np


def init():
    trainfile_X = r'E:\Machine Learning\MLData\mnist\train-images.idx3-ubyte'
    trainfile_y = r'E:\Machine Learning\MLData\mnist\train-labels.idx1-ubyte'
    testfile_X = r'E:\Machine Learning\MLData\mnist\t10k-images.idx3-ubyte'
    testfile_y = r'E:\Machine Learning\MLData\mnist\t10k-labels.idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()
    ss = StandardScaler()
    train_X = ss.fit_transform(train_X)
    test_X = ss.transform(test_X)
    return train_X, train_y, test_X, test_y


def LSVC(train_X, train_y, test_X, test_y):
    lsvc = LinearSVC()
    lsvc.fit(train_X, train_y)
    predict_y = lsvc.predict(test_X)

    print(lsvc.score(test_X, test_y))
    print(classification_report(test_y, predict_y))


def PrincipalComponentAnalysis(train_X, train_y, test_X, test_y):
    estimator = PCA(n_components=3)

    train_X_pca = estimator.fit_transform(train_X)
    test_X_pca = estimator.transform(test_X)
    lsvc = LinearSVC()
    lsvc.fit(train_X_pca, train_y)
    predict_y = lsvc.predict(test_X_pca)

    print(lsvc.score(test_X_pca, test_y))
    print(classification_report(test_y, predict_y))


def main():
    train_X, train_y, test_X, test_y = init()
    print(train_X.shape[0])
    LSVC(train_X, train_y, test_X, test_y)
    PrincipalComponentAnalysis(train_X, train_y, test_X, test_y)


if __name__ == "__main__":
    main()
