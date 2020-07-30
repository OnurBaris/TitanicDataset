# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:25:26 2020

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

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score
from sklearn.metrics import confusion_matrix, f1_score

param_grid = {
    'n_estimators' : [50, 100, 150, 200, 250, 300, 400, 500],
    'max_features' : ['auto', 'sqrt'],
    'max_depth': [None, 5, 10, 15, 20, 25, 30, 40, 50],
    'min_samples_split': [3, 4, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score)
}


def grid_search_wrapper(refit_score='f1_score'):

    """fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics"""
                      
    grid_search = GridSearchCV(model, param_grid, scoring=scorers, refit=refit_score, cv=10, return_train_score=True, n_jobs=-1)

    grid_search.fit(X, y)

    # make the predictions
    labels_pred = grid_search.predict(X)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y, labels_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search

grid_search_clf = grid_search_wrapper()