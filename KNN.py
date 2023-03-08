#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression , BayesianRidge , ARDRegression
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
import pickle

dtf = pd.read_csv('dataset.csv')

Data = dtf.drop('class' ,  axis=1)
#Data = Data.drop('x12',axis=1)
#Data = Data.drop('y12',axis=1)
#Data = Data.drop('x7', axis=1)
#Data = Data.drop('y7',axis=1)
#Data = Data.drop('x9', axis=1)
#Data = Data.drop('y9',axis=1)
#Data = Data.drop('x10',axis=1)
#Data = Data.drop('y10' , axis=1)
#Data = Data.drop('x6',axis=1)
#Data = Data.drop('y6' , axis=1)



Target = dtf['class']

#labelencoder = LabelEncoder()
#Target = labelencoder.fit_transform(Target)
print(Data)
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
#Data = NormalizeData(np.array(Data))
print(Data)
print(Target.value_counts())

X_train, X_test, y_train, y_test = train_test_split(Data, Target,test_size=0.3, random_state=100)
under_sampler = RandomUnderSampler(random_state=42)
X_train, y_train = under_sampler.fit_resample(X_train, y_train)
X_test, y_test = under_sampler.fit_resample(X_test, y_test)

knn = KNeighborsClassifier(n_neighbors=50,algorithm="ball_tree",leaf_size=25 ,
                           weights='distance',metric='minkowski')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
#y_pred = [round(value) for value in y_pred]
print('Accuracy of KNN after tuning',accuracy_score(y_pred,y_test))
print(confusion_matrix(y_test, y_pred))

#with open('knn_all_exe.pkl', 'wb') as f:
   #pickle.dump(knn, f)


