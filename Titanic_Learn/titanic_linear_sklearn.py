# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 19:32:41 2017

@author: thinkwee
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

vali = pd.read_csv(r"E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\validation.csv")
train = pd.read_csv(r"E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\train.csv")
train = train.dropna(subset=['Age', 'Embarked'], axis=0)
vali = vali.dropna(subset=(['Age', 'Embarked']), axis=0)

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
trainx = train.reindex(index=train.index[:],
                       columns=['Age'] + ['Sex'] + ['Parch'] + ['Fare'] + ['Embarked'] + ['SibSp'])

vali.loc[vali["Sex"] == "male", "Sex"] = 0
vali.loc[vali["Sex"] == "female", "Sex"] = 1
vali.loc[vali["Embarked"] == "S", "Embarked"] = 0
vali.loc[vali["Embarked"] == "C", "Embarked"] = 1
vali.loc[vali["Embarked"] == "Q", "Embarked"] = 2
vali1 = vali.reindex(index=vali.index[:], columns=['Age'] + ['Sex'] + ['Parch'] + ['Fare'] + ['Embarked'] + ['SibSp'])

survive = vali.reindex(index=vali.index[:], columns=['Survived'])
survive = np.array(survive)

feature = ['Age', 'Sex', 'Parch', 'Fare', 'Embarked', 'SibSp']

trainy = train.reindex(index=train.index[:], columns=['Survived'])
trainy = trainy.Survived

X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, random_state=1)

model = LinearRegression()
print(X_train)
model.fit(X_train, y_train)
print(model)

for i in zip(feature, model.coef_):
    print(i)

predict = model.predict(vali1)

count = 0
for i in range(len(predict)):
    if (predict[i] > 1 and survive[i] == 1) or (predict[i] < 1 and survive[i] == 0):
        count = count + 1.0

print("prediction finish")
print("prediction ratio:", "%f %%" % (count / len(predict) * 100))
