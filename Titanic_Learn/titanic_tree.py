# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:27:08 2017

@author: thinkwee
"""

import csv as csv 
import pandas as pd
import numpy as np

test_file=(open(r'E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\test.csv', 'r'))
test_file_object = csv.reader(open(r'E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\test.csv', 'r'))
testheader = next(test_file_object)#test set


train = pd.read_csv(r"E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\train.csv") 

train.loc[train["Sex"]=="male","Sex"]=0
train.loc[train["Sex"]=="female","Sex"]=1         
         
train["Embarked"] = train["Embarked"].fillna("S")
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
         



