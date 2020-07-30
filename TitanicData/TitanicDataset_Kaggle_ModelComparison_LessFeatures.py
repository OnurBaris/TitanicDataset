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
train_data.head()

test_data = pd.read_csv("C:/01_Projects/09_CriticalFormulasandTools/PythonScripts/TitanicData/test.csv")
test_data.head()

y = train_data["Survived"]

features = ["Pclass", "Sex", "Fare", "Age"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
X_test = my_imputer.fit_transform(X_test)

model1 = GaussianNB()
model1.fit(X, y)
model2 = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=1)
model2.fit(X, y)
model3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)
model3.fit(X, y)
model4 = KNeighborsClassifier(3)
model4.fit(X, y)

labels_pred1 = model1.predict(X)
labels_pred2 = model2.predict(X)
labels_pred3 = model3.predict(X)
labels_pred4 = model4.predict(X)

print(pd.DataFrame(confusion_matrix(y, labels_pred1),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y, labels_pred2),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y, labels_pred3),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
print(pd.DataFrame(confusion_matrix(y, labels_pred4),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

predictions = model2.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")