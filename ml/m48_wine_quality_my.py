import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
score = pd.read_csv('../data/csv/data-01-test-score.csv')
wine = pd.read_csv('../data/csv/winequality-white.csv', sep = ';')

print(score.shape)          # (24, 4)
print(wine.shape)           # (4898, 12)
print(wine.columns)
# Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
#        'pH', 'sulphates', 'alcohol', 'quality'],
#       dtype='object')

# 2. Split Data
x = wine.iloc[:, :-1].values
y = wine.iloc[:, -1].values
print(x.shape, y.shape)     # (4898, 11) (4898,)
# sns.countplot(y)
# plt.show()
# y값: 3, 4, 5, 6, 7, 8, 9

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)


# 3. SelectFromModel
parameters = [
    {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.7, 0.9]}
]

model = RandomizedSearchCV(XGBClassifier(n_jobs = -1), parameters, cv = 5)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('BEST_ESTIMATOR: \n', model.best_estimator_)
print('FEATURE_IMPORTANCES: \n', model.best_estimator_.feature_importances_)
print('SCORE: \n', score)
# BEST_ESTIMATOR: 
#  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.1, max_delta_step=0, max_depth=6,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=110, n_jobs=-1, num_parallel_tree=1,
#               objective='multi:softprob', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# FEATURE_IMPORTANCES: 
#  [0.06856259 0.1135421  0.07630071 0.08350154 0.07662345 0.08628306
#  0.0712667  0.09684834 0.07337713 0.07008377 0.18361062]
# SCORE:
#  0.6428571428571429

thresholds = np.sort(model.best_estimator_.feature_importances_)
for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)
    select_x_train = selection.transform(x_train)

    selection_model = XGBClassifier(n_jobs = -1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = accuracy_score(y_test, y_predict)

    print('Thresh = %.3f, n = %d, ACC: %.2f%%' %(thresh, select_x_train.shape[1],
          score*100))
# Thresh = 0.069, n = 11, ACC: 67.45%

# 4. Result
select = SelectFromModel(model.best_estimator_, threshold = 0.069, prefit = True)
select_x_train = select.transform(x_train)
select_x_test = select.transform(x_test)

# 4-1. XGBClassifier
parameters = [
    {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.7, 0.9]}
]

grid_model = GridSearchCV(XGBClassifier(n_jobs = -1), parameters, cv = 5)
grid_model.fit(select_x_train, y_train)
score = grid_model.score(select_x_test, y_test)
print('########RESULT########')
print('Thresh = %.3f, n = %d, ACC: %.2f%%' %(0.069, select_x_train.shape[1], score*100))
