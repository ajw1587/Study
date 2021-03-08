from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

print('__________')

x = load_boston().data
y = load_boston().target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)

parameters = [
    {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.7, 0.9]}
]

model = RandomizedSearchCV(XGBRegressor(n_jobs = -1), parameters, cv = 5)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
thresholds = model.best_estimator_.feature_importances_
print('BEST_PARAMETERS: \n', model.best_estimator_)
print('FEATURE_IMPORTANCES: \n', thresholds)
print('SCORE: ', score)

for thresd in thresholds:
    selection = SelectFromModel(model, threshold = thresd, prefit = True)
    select_x_train = selection.transform(x_train)
    
