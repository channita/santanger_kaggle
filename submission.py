# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:17:59 2016

@author: Channita
"""
print("IMPORTING LIBRARIES...")

import pandas as pd
import numpy as np
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import cross_validation
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from collections import OrderedDict

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.cross_validation import KFold
#%%

print("LOADING DATA...")

#cd C:\Users\Channita\Desktop\CHANNITA\Study\Term 3\Analytics in financial services\Final

train = pd.read_csv("train.csv")
train.head()

test = pd.read_csv("test.csv")
test.head()

y = train['TARGET']

X = train.drop(['ID','TARGET'],axis=1)

#%%

print("PREPROCESSING...")

#Cleaning

## Removing duplicates
duplicates = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            duplicates.append(columns[j])

## Removing dirty columns

droplist =  ['imp_amort_var18_hace3', 'imp_amort_var34_hace3', 'imp_reemb_var13_hace3', 'imp_reemb_var33_hace3',
                     'imp_trasp_var17_out_hace3', 'imp_trasp_var33_out_hace3', 'ind_var2', 'ind_var27', 'ind_var27_0',
                     'ind_var28', 'ind_var28_0', 'ind_var2_0', 'ind_var41', 'ind_var46', 'ind_var46_0',
                     'num_reemb_var13_hace3', 'num_reemb_var33_hace3', 'num_trasp_var17_out_hace3',
                     'num_trasp_var33_out_hace3', 'num_var27', 'num_var27_0', 'num_var28', 'num_var28_0',
                     'num_var2_0_ult1', 'num_var2_ult1', 'num_var41', 'num_var46', 'num_var46_0',
                     'saldo_medio_var13_medio_hace3', 'saldo_var27', 'saldo_var28', 'saldo_var2_ult1', 'saldo_var41',
                     'saldo_var46', 'delta_num_reemb_var13_1y3', 'delta_num_reemb_var17_1y3',
                     'delta_num_reemb_var33_1y3', 'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3',
                     'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3', 'ind_var13_medio_0',
                     'ind_var18_0', 'ind_var25_0', 'ind_var26_0', 'ind_var6', 'ind_var6_0', 'ind_var32_0',
                     'ind_var34_0', 'ind_var37_0', 'ind_var40', 'num_var13_medio_0', 'num_var18_0', 'num_var25_0',
                     'num_var26_0', 'num_var6', 'num_var6_0', 'num_var32_0', 'num_var34_0', 'num_var37_0',
                     'num_var40', 'saldo_var13_medio', 'saldo_var6']

X = X.drop(duplicates + droplist, axis=1, inplace = False)

X_test = test.drop(duplicates + droplist + ["ID"], axis=1, inplace = False)

#from sklearn.preprocessing import Imputer

#imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
#imp.fit(Xo)
#new_Xo=imp.transform(Xo)

# Preprocessing
X_scaled = StandardScaler().fit_transform(X)

# PCA
pca_fit = PCA().fit(X_scaled)
pca_features = pca_fit.transform(X_scaled)

#%%

print("EXPLORATORY DATA ANALYSIS...")

## Only 3% of data base are unhappy 
train_happy = train[train['TARGET'] == 0]
train_unhappy = train[train['TARGET'] == 1]

## Describing any variable 

train.var15.describe()
#var15 is suspected to be the age of the customer
#show distribution in histogram chart
train.var15.hist(bins=100) 
plt.show()

# Top-10 most common values
train.var3.value_counts()[:10]

#%%

#Random forest
startTime = time.time()

rf = RandomForestRegressor(n_estimators = 10, max_features = 'log2', oob_score = False, random_state = 0, warm_start = True)
result = rf.fit(X_scaled, y)

print(result.score(X_scaled, y))
# scores = cross_val_score(rf, X, y)

timeToTrain = (time.time()-startTime)/60
print(timeToTrain)

y_pred = result.predict(X_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred})

submission.to_csv("submission.csv", index=False)

#%%

print("FINDING IMPORTANT FEATURES ")

importances = result.feature_importances_
std = np.std([tree.feature_importances_ for tree in result.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

list_inputs = train.columns

for f in range(284):
    print(f + 1, list_inputs[indices[f]], importances[indices[f]])
    
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_scaled.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_scaled.shape[1]), indices)
plt.xlim([-1, X_scaled.shape[1]])
plt.show()

