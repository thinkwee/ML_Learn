# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 10:33:54 2016

using multivaraite linear regression to solve the Titanic Problem on Kaggle

@author: thinkwee
"""
import csv as csv 
import numpy as np
import pandas as pd

test_file=(open(r'E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\test.csv', 'r'))
test_file_object = csv.reader(open(r'E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\test.csv', 'r'))
testheader = next(test_file_object)#test set

train=pd.read_csv(r"E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\train.csv")

vali=pd.read_csv(r"E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\validation.csv")

valipredict_file_object=csv.writer(open(r'E:\Machine Learning\MLData\Titanic Machine Learning from Disaster\valipredict.csv', 'w')) 
#validation result set

train1=train.dropna(subset=(['Age']),axis=0)
vali1=vali.dropna(subset=(['Age']),axis=0)

validata=np.array(vali1)
data=np.array(train1)

fare_ceiling = 40
data[data[0::,9].astype(np.float)>=fare_ceiling,9] = fare_ceiling - 1.0

train = np.column_stack((data[0::,9],data[0::,2],data[0::,5],data[0::,4]))
predict=np.column_stack((validata[0::,9],validata[0::,2],validata[0::,5],validata[0::,4]))
survive = np.column_stack((data[0::,1]))


for i in range(train.shape[0]):
    if (train[i][3]=='male'):
        train[i][3]=0.00
    else:
        train[i][3]=1.00
for i in range(predict.shape[0]):
    if (predict[i][3]=='male'):
        predict[i][3]=0.00
    else:
        predict[i][3]=1.00


x0=np.ones((train.shape[0],1))
train=np.concatenate((train,x0),axis=1)

x0=np.ones((predict.shape[0],1))
predict=np.concatenate((predict,x0),axis=1)

print('raw data finish')

survive=survive.T.astype(np.float)
traint=train.T.astype(np.float)
w0=traint.dot(train.astype(np.float))
w1=(np.linalg.inv(w0))  
w2=w1.dot(traint)
w=w2.dot(survive)  #w=(Xt*X)^-1*Xt*y
print('w calc finish')

feature=['Fare','Pclass','Age','Sex','b']
for i in zip(feature,w):
    print(i)


valipredict_file_object.writerow(["PassengerName", "Actual Survived","Predict Survived","XO"])
count=0.0
for i in range(predict.shape[0]):
    temp=predict[i,0::].T.astype(float)
    answer=temp.dot(w)
    answer=answer[0]
    if ((answer>0.5 and validata[i][1]==1) or (answer<0.5 and validata[i][1]==0)):
        flag="Correct"
        count=count+1.0;
    else:
        flag="Error"
    valipredict_file_object.writerow([validata[i][3],validata[i][1],answer,flag])

print("prediction finish")
print("prediction ratio:","%f %%"%(count/predict.shape[0]*100))



