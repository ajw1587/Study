# diabetes
# 0.5 이상

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import numpy as np

x = load_diabetes().data
y = load_diabetes().target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)

parameters = [
    {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.7, 0.9]}
]
model = RandomizedSearchCV(XGBRegressor(n_jobs = -1), parameters, cv = 5)
model.fit(x_train, y_train)

thresholds = np.sort(model.best_estimator_.feature_importances_)
score = model.score(x_test, y_test)
print('BEST_ESTIMATOR: \n', model.best_estimator_)
print('FEATURE_IMPORTANCES: \n', thresholds)
print('SCORE: ', score)

for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)

    model2 = XGBRegressor(n_jobs = -1)
    model2.fit(select_x_train, y_train)

    score = model2.score(select_x_test, y_test)

    print('THRESHOLD: {:.6f}   n = {:d}  SCORE: {:.6f}'.format(
        thresh, select_x_train.shape[1], score
    ))

print('\n###################################################\n')
###################################################

selection = SelectFromModel(model.best_estimator_, threshold = 0.031814, prefit = True)
select_x_train = selection.transform(x_train)
select_x_test = selection.transform(x_test)

model = GridSearchCV(model.best_estimator_, parameters, cv = 5)
model.fit(select_x_train, y_train)
score = model.score(select_x_test, y_test)
print('Thresh = %.3f, n = %d, R2: %.2f%%' %(0.031814, select_x_train.shape[1], score*100))