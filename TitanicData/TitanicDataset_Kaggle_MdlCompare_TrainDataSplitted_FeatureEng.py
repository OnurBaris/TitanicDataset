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
        
train_data = pd.read_csv("C:/01_Projects/09_CriticalFormulasandTools/PythonScripts/TitanicData/train.csv")
test_data = pd.read_csv("C:/01_Projects/09_CriticalFormulasandTools/PythonScripts/TitanicData/test.csv")

y = train_data["Survived"]

train_data["Title"] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data["Title"] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data["familysize"]=train_data["SibSp"]+train_data["Parch"]+1
test_data["familysize"]=test_data["SibSp"]+test_data["Parch"]+1

features = ["Pclass", "Sex", "Fare", "Title", "familysize", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

zero_array = np.zeros(418)
zero_array2 = np.zeros(891)

X_test.loc[:,'Title_Capt']=zero_array
X_test.loc[:,'Title_Countess']=zero_array
X_test.loc[:,'Title_Don']=zero_array
X.loc[:,'Title_Dona']=zero_array2
X_test.loc[:,'Title_Jonkheer']=zero_array
X_test.loc[:,'Title_Lady']=zero_array
X_test.loc[:,'Title_Major']=zero_array
X_test.loc[:,'Title_Mlle']=zero_array
X_test.loc[:,'Title_Mme']=zero_array
X_test.loc[:,'Title_Sir']=zero_array

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
X_test = my_imputer.fit_transform(X_test)

X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = GaussianNB()
model1.fit(X_train, y_train)
model2 = RandomForestClassifier(max_depth=20, n_estimators=100, bootstrap=True, max_features= 'sqrt', min_samples_leaf=1, min_samples_split=20)
model2.fit(X_train, y_train)
model3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)
model3.fit(X_train, y_train)
model4 = KNeighborsClassifier(3)
model4.fit(X_train, y_train)
model5 = DecisionTreeClassifier(random_state=1)
model5.fit(X_train, y_train)

labels_pred1 = model1.predict(X_test1)
labels_pred2 = model2.predict(X_test1)
labels_pred3 = model3.predict(X_test1)
labels_pred4 = model4.predict(X_test1)
labels_pred5 = model5.predict(X_test1)

labels_pred6 = model1.predict(X_test)
labels_pred7 = model2.predict(X_test)
labels_pred8 = model3.predict(X_test)
labels_pred9 = model4.predict(X_test)
labels_pred10 = model5.predict(X_test)

print(pd.DataFrame(confusion_matrix(y_test1, labels_pred1),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y_test1, labels_pred2),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y_test1, labels_pred3),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y_test1, labels_pred4),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y_test1, labels_pred5),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

predictions = model2.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")