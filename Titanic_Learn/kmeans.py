# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:08:42 2017

@author: thinkwee
"""
import pandas as pd
import scipy
import numpy as np
import csv as csv
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


def init():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.fit_transform(X_test)
    return X_train, X_test, y_train, y_test, iris


def initk(X_train, k):
    C = [X_train[0]]
    for i in range(1, k):
        D2 = scipy.array([min([scipy.inner(c - x, c - x) for c in C]) for x in X_train])
        probs = D2 / D2.sum()
        cumprobs = probs.cumsum()
        r = scipy.rand()
        for j, p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X_train[i])
    return C


def evaluate(C, X_train, y_predict):
    sum = 0
    for i in range(len(X_train)):
        c = C[y_predict[i]]
        sum += scipy.inner(c - X_train[i], c - X_train[i])
    return sum


def cluster(C, X_train, y_predict, k):
    sum = [0, 0, 0, 0] * k
    count = [0] * k
    newC = []
    for i in range(len(X_train)):
        min = 32768
        minj = -1
        for j in range(k):
            if scipy.inner(C[j] - X_train[i], C[j] - X_train[i]) < min:
                min = scipy.inner(C[j] - X_train[i], C[j] - X_train[i])
                minj = j
        y_predict[i] = (minj + 1) % k
    for i in range(len(X_train)):
        sum[y_predict[i]] += X_train[i]
        count[y_predict[i]] += 1
    for i in range(k):
        newC.append(sum[i] / count[i])
    return y_predict, newC


def main():
    X_train, X_test, y_train, y_test, iris = init()
    k = 3
    total = len(y_train)
    y_predict = [0] * total
    C = initk(X_train, k)
    oldeval = evaluate(C, X_train, y_predict)
    while (1):
        y_predict, C = cluster(C, X_train, y_predict, k)
        neweval = evaluate(C, X_train, y_predict)
        ratio = (oldeval - neweval) / oldeval * 100
        print(oldeval, " -> ", neweval, "%f %%" % ratio)
        oldeval = neweval
        if ratio < 0.1:
            break

    print(y_train)
    print(y_predict)
    n = 0
    m = 0
    for i in range(len(y_train)):
        m += 1
        if y_train[i] == y_predict[i]:
            n += 1
    print(n / m)
    print(classification_report(y_train, y_predict, target_names=iris.target_names))


    #
    # p=csv.writer(open(r'E:\Machine Learning\MLData\iris-species\predict.csv', 'w',newline=''))
    # for i in range(len(L)-1):
    #     p.writerow([X_train[i][0],X_train[i][1],X_train[i][2],X_train[i][3],y_predict[i]])


if __name__ == "__main__":
    main()
