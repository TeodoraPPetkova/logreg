# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 11:55:10 2020

@author: user
"""
# GRID SEARCH
# ADABOOST

# Raiffeisen Bank Case Study
# Path-2-Digital
# MOBILE BANKING

# Load libraries
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

# Set output display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.6f}'.format

# Import data
FINAL = pd.read_csv("MB_FINAL.csv")

# Separate Variables
X=FINAL.drop(['client_id','month','FLAG_new_active_user_MB'], axis=1) # predictors
y=FINAL.iloc[:,2] # target

# split the dataset for each month
nov=FINAL.loc[FINAL['month']==201911]
dec=FINAL.loc[FINAL['month']==201912]
jan=FINAL.loc[FINAL['month']==202001]
feb=FINAL.loc[FINAL['month']==202002]
mar=FINAL.loc[FINAL['month']==202003]
apr=FINAL.loc[FINAL['month']==202004]
may=FINAL.loc[FINAL['month']==202005]
jun=FINAL.loc[FINAL['month']==202006]
jul=FINAL.loc[FINAL['month']==202007]

# Form train and test sets
train=pd.concat([nov,dec,feb,mar,may,jun])
test=pd.concat([jan,apr,jul])

# Predictors and target
X_train=train.drop(['client_id','month','FLAG_new_active_user_MB'], axis=1) # predictors
y_train=train.iloc[:,2] # target

X_test=test.drop(['client_id','month','FLAG_new_active_user_MB'], axis=1) # predictors
y_test=test.iloc[:,2] # target

# Model
model = AdaBoostClassifier()
# define the grid of values to search
grid = dict()
grid['n_estimators'] = [200, 500]
grid['learning_rate'] = [0.01, 0.1, 1.0, 1.5]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
# execute the grid search
grid_result = grid_search.fit(X_train, y_train)
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
