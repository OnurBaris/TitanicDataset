# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:49:37 2020

@author: obaris
"""

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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
train_data = pd.read_csv("C:/01_Dosyalar/PythonFiles/Codes/TitanicData/train.csv")
test_data = pd.read_csv("C:/01_Dosyalar/PythonFiles/Codes/TitanicData/test.csv")

y = train_data["Survived"]

train_data["Title"] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data["Title"] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data["familysize"]=train_data["SibSp"]+train_data["Parch"]+1
test_data["familysize"]=test_data["SibSp"]+test_data["Parch"]+1
train_data.loc[(train_data['Age']<=16),'Age']=1
train_data.loc[(train_data['Age']>16) & (train_data['Age']<=32),'Age']=2
train_data.loc[(train_data['Age']>32) & (train_data['Age']<=48),'Age']=3
train_data.loc[(train_data['Age']>48) & (train_data['Age']<=64),'Age']=4
train_data.loc[(train_data['Age']>64) & (train_data['Age']<=80.),'Age']=5
test_data.loc[(test_data['Age']<=16),'Age']=1
test_data.loc[(test_data['Age']>16) & (test_data['Age']<=32),'Age']=2
test_data.loc[(test_data['Age']>32) & (test_data['Age']<=48),'Age']=3
test_data.loc[(test_data['Age']>48) & (test_data['Age']<=64),'Age']=4
test_data.loc[(test_data['Age']>64) & (test_data['Age']<=80.),'Age']=5

features = ["Pclass", "Sex", "Fare", "Title", "familysize", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

print(X.head())

X_test['Title_Capt']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Countess']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Don']=X_test.apply(lambda x: 0, axis=1)
X['Title_Dona']=X.apply(lambda x: 0, axis=1)
X_test['Title_Jonkheer']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Lady']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Major']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Mlle']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Mme']=X_test.apply(lambda x: 0, axis=1)
X_test['Title_Sir']=X_test.apply(lambda x: 0, axis=1)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

scaler=StandardScaler()
X_scaled = X
X_scaled_test = X_test
X_scaled[['Pclass','Fare','familysize','Age']] = scaler.fit_transform(X_scaled[['Pclass','Fare','familysize','Age']])
X_scaled_test[['Pclass','Fare','familysize','Age']] = scaler.fit_transform(X_scaled_test[['Pclass','Fare','familysize','Age']])

X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())
X_scaled = X_scaled.fillna(X_scaled.median())
X_scaled_test = X_scaled_test.fillna(X_scaled_test.median())

model1 = GaussianNB()
model1.fit(X, y)
model2 = RandomForestClassifier(max_depth=20, n_estimators=100, bootstrap=True, max_features= 'sqrt', min_samples_leaf=1, min_samples_split=20)
model2.fit(X, y)
model3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(20,5), random_state=1, max_iter=3000)
model3.fit(X_scaled, y)
model4 = KNeighborsClassifier(3)
model4.fit(X, y)
model5 = DecisionTreeClassifier(random_state=1)
model5.fit(X, y)

labels_pred1 = model1.predict(X)
labels_pred2 = model2.predict(X)
labels_pred3 = model3.predict(X_scaled)
labels_pred4 = model4.predict(X)
labels_pred5 = model5.predict(X)

labels_pred6 = model1.predict(X_test)
labels_pred7 = model2.predict(X_test)
labels_pred8 = model3.predict(X_scaled_test)
labels_pred9 = model4.predict(X_test)
labels_pred10 = model5.predict(X_test)

print(pd.DataFrame(confusion_matrix(y, labels_pred1), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y, labels_pred2), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y, labels_pred3), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y, labels_pred4), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y, labels_pred5), columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

predictions = model3.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")