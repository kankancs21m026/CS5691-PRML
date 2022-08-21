#!/usr/bin/env python
# coding: utf-8

# In[59]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # PACKAGES 

# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt


# # PROCESSING THE DATA TO GET IN THE REQUIRED FORMATE

# In[61]:


#reading the dataset1
train1=pd.read_csv("Dataset_1_Training.csv")
test1=pd.read_csv("Dataset_1_Testing.csv")
#processing the dataset1 to get in required formate
train1=train1.T
train1.columns = train1.iloc[0] 
train1=train1[1:]
y1=train1['CO: 1']
y2=train1['CO: 2']
train1=train1.drop(['CO: 1','CO: 2'],axis=1)
y1=y1.astype(int)
y2=y2.astype(int)
test1=test1.T
test1.columns = test1.iloc[0] 
test1=test1[1:]
#reading the dataset2
train2=pd.read_csv("Dataset_2_Training.csv")
test2=pd.read_csv("Dataset_2_Testing.csv")
#processing the dataset2 to get in required formate
train2=train2.T
train2.columns = train2.iloc[0] 
train2=train2[1:]
y3=train2['CO: 3']
y4=train2['CO: 4']
y5=train2['CO: 5']
y6=train2['CO: 6']
train2=train2.drop(['CO: 3','CO: 4','CO: 5','CO: 6'],axis=1)
y3=y3.astype(int)
y4=y4.astype(int)
y5=y5.astype(int)
y6=y6.astype(int)
test2=test2.T
test2.columns = test2.iloc[0] 
test2=test2[1:]


# # SETTING THE RANDOM SEED IN ORDER TO GET THE SAME RESULT EVERYTIME

# # PCA CURVE FOR DATASET1

# In[65]:


'''from sklearn import decomposition
pca=PCA()

pca_data = pca.fit_transform(train1)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show() '''


# # PCA CURVE FOR DATASET2

# In[66]:


'''from sklearn import decomposition
pca=PCA()

pca_data = pca.fit_transform(train1)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show() '''


# In[67]:


def setSeed():
    np.random.seed(0)
setSeed()


# # **separating the datasets into training and testing in the ratio of 80:20 and setting random_state equal to 10 to get the same split everytime**

# In[68]:


X_train1,X_test1,y1_train,y1_test=train_test_split(train1,y1,test_size=.2,random_state=10)
X_train1,X_test1,y2_train,y2_test=train_test_split(train1,y2,test_size=.2,random_state=10)
X_train2,X_test2,y3_train,y3_test=train_test_split(train2,y3,test_size=.2,random_state=10)
X_train2,X_test2,y4_train,y4_test=train_test_split(train2,y4,test_size=.2,random_state=10)
X_train2,X_test2,y5_train,y5_test=train_test_split(train2,y5,test_size=.2,random_state=10)
X_train2,X_test2,y6_train,y6_test=train_test_split(train2,y6,test_size=.2,random_state=10)


# # SUPERVISED ALGORITHAMS FOR EACH CLINICAL DESCRIPTOR

# In[ ]:


"""Logistic_grid={'C':[1e-3,1e-2, 1e-1, 1, 10],'penalty':['l1', 'l2']}
Logistic_estimator=LogisticRegression(solver='liblinear',random_state=0)
Logistic_model1=GridSearchCV(estimator=Logistic_estimator, param_grid=Logistic_grid,cv=3,scoring='accuracy')
Logistic_model1.fit(X_train1,y1_train)
print("Best Parameters are ;",Logistic_model1.best_params_)
y1_p=Logistic_model1.predict(X_test1)
accuracy=accuracy_score(y1_test,y1_p)
print('Logistic Regression giving the accuracy ;',accuracy)
# By doing parameter tuning using grid search CV, the best parameter are: Best Parameters are ; {'C': 0.001, 'penalty': 'l2'} """


# In[69]:


Logistic_model1=LogisticRegression(solver='liblinear',random_state=0,C= 0.001,penalty='l2')
Logistic_model1.fit(train1,y1)
y1_pred=Logistic_model1.predict(test1)


# In[ ]:


"""Logistic_grid={'C': [1e-03, 1e-2, 1e-1, 1, 10],'penalty': ['l1', 'l2']}
Logistic_estimator=LogisticRegression(solver='liblinear',random_state=0)
Logistic_model2=GridSearchCV(estimator=Logistic_estimator,param_grid=Logistic_grid,cv=3,scoring='accuracy')
Logistic_model2.fit(X_train1,y2_train) #for X_train1 C=1 and l1
print("Best Parameters are ;",Logistic_model2.best_params_)
y2_p=Logistic_model2.predict(X_test1)
accuracy=accuracy_score(y2_test,y2_p)
print('Logistic Regression accuracy:',accuracy)
#By doing parameter tuning using grid search CV, the best parameter are: Best Parameters are ; {'C': 1, 'penalty': 'l1'}  """


# In[70]:


train1_pca=train1
train2_pca=train2
test1_pca=test1
test2_pca=test2


# # PCA

# In[71]:


from sklearn.decomposition import PCA
pca=PCA()
pca.fit(train1_pca)
train1_pca=pca.transform(train1_pca)
test1_pca=pca.transform(test1_pca)
pca=PCA()
pca.fit(train2_pca)
train2_pca=pca.transform(train2_pca)
test2_pca=pca.transform(test2_pca)


# In[72]:


Logistic_model2=LogisticRegression(solver='liblinear',random_state=0,C=10,penalty='l2')
Logistic_model2.fit(train1_pca,y2)
y2_pred=Logistic_model2.predict(test1_pca)


# In[ ]:


"""Logistic_grid={'C':[1e-03,1e-2,1e-1,1,10],'penalty':['l1','l2']}
Logistic_estimator=LogisticRegression(solver='liblinear',random_state=0)
Logistic_model3=GridSearchCV(estimator=Logistic_estimator,param_grid=Logistic_grid,cv=3,scoring='accuracy')
Logistic_model3.fit(train2,y3)
print("Best Parameters are ;",Logistic_model3.best_params_)
#y3_p=Logistic_model3.predict(X_test2)
#accuracy=accuracy_score(y3_test,y3_p)
#print('Logistic Regression accuracy:',accuracy) """


# In[73]:


Logistic_model3=LogisticRegression(solver='liblinear',random_state=0,C=0.1,penalty='l1')
Logistic_model3.fit(train2,y3)
y3_pred=Logistic_model3.predict(test2)


# In[ ]:


"""Logistic_grid={'C':[1e-03, 1e-2, 1e-1, 1, 10],'penalty':['l1', 'l2']}
Logistic_estimator=LogisticRegression(solver='liblinear',random_state=0)
Logistic_model4=GridSearchCV(estimator=Logistic_estimator,param_grid=Logistic_grid,cv=3,scoring='accuracy')
Logistic_model4.fit(train2,y4)
print("Best Parameters:\n", Logistic_model4.best_params_)
#y4_p=Logistic_model4.predict(X_test2)
#accuracy=accuracy_score(y4_test,y4_p)
#print('Logistic Regression accuracy are ;',accuracy) """


# In[74]:


Logistic_model4=LogisticRegression(solver='liblinear',random_state=0,C=0.001,penalty='l1')
Logistic_model4.fit(train2,y4)
y4_pred=Logistic_model4.predict(test2)


# In[ ]:


"""Logistic_grid={'C':[1e-03, 1e-2, 1e-1, 1, 10],'penalty':['l1', 'l2']}
Logistic_estimator=LogisticRegression(solver='liblinear',random_state=0)
Logistic_model5=GridSearchCV(estimator=Logistic_estimator,param_grid=Logistic_grid,cv=3,scoring='accuracy')
Logistic_model5.fit(train2,y5)
print("Best Parameters:\n", Logistic_model5.best_params_)
#y4_p=Logistic_model4.predict(X_test2)
#accuracy=accuracy_score(y4_test,y4_p)
#print('Logistic Regression accuracy are ;',accuracy) """


# In[76]:


Logistic_model5=LogisticRegression(solver='liblinear',random_state=0,C=0.01,penalty='l1')
Logistic_model5.fit(train2,y5)
y5_pred=Logistic_model5.predict(test2)


# In[ ]:


param_grid={'n_estimators':[10, 20, 30]}
estimator=AdaBoostClassifier(DecisionTreeClassifier(random_state=0,max_depth=1))
adaBoost_model=GridSearchCV(estimator,param_grid,cv=3,scoring='accuracy')
adaBoost_model=adaBoost_model.fit(X_train2,y6_train)
#print("Best Parameters are;",adaBoost_model.best_params_)
#y_p=adaBoost_model.predict(X_test2)
#accuracy=accuracy_score(y6_test,y_p)
#print('AdaBoostClassifier model accuracy ;',accuracy) 


# In[77]:


#adaBoost_model6=AdaBoostClassifier(n_estimators=10)
#adaBoost_model6.fit(train2,y6)
y6_pred=adaBoost_model.predict(test2)


# # COMBINING ALL THE PREDICTORS INTO ONE FILE prediction.csv

# In[78]:


temp=np.concatenate((y1_pred, y2_pred, y3_pred, y4_pred, y5_pred, y6_pred), axis=None)
pd.DataFrame(temp, columns=['Predicted']).to_csv('prediction.csv',index_label='Id')


# 
