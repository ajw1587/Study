# 실습
# 1. 상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성
# 최적의 R2값과 피처임포턴스 구할것

# 2. 위 쓰레드 값으로 SelectFrom을 구해서
# 최적의 피처 갯수를 구할것

# 3. 위 피처 갯수로 데이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용하여
# 최적의 R2 구할것

# 1번과 2번값 비교

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import numpy as np

dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, train_size = 0.8, shuffle = True, random_state = 77)

# 1번
parameters = [
    {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.7, 0.9]}
]
model = RandomizedSearchCV(XGBRegressor(n_jobs = -1), parameters, cv = 5)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print(model.best_estimator_.feature_importances_)
# [0.02066543 0.00363883 0.01179045 0.01456683 0.06878352 0.36984253
#  0.00920059 0.05361473 0.01084579 0.02106604 0.03052855 0.00863272
#  0.376824  ]

print('Best Parameters: ', model.best_estimator_)
# base_score=0.5, booster='gbtree', colsample_bylevel=1,
# colsample_bynode=1, colsample_bytree=0.9, gamma=0, gpu_id=-1,
# importance_type='gain', interaction_constraints='',
# learning_rate=0.1, max_delta_step=0, max_depth=5,
# min_child_weight=1, missing=nan, monotone_constraints='()',
# n_estimators=110, n_jobs=-1, num_parallel_tree=1, random_state=0,
# reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
# tree_method='exact', validate_parameters=1, verbosity=None

print('R2 SCORE: ', score)
# R2 SCORE:  0.9125620494944608

thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds)
# [0.00170744 0.00860656 0.00981789 0.01022    0.02171908 0.02928848
#  0.03224914 0.04784627 0.05713698 0.06915355 0.12243687 0.2697346
#  0.32008317]

for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)

    selection_model = XGBRegressor(n_jobs = -1)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh = %.3f, n = %d, R2: %.2f%%' %(thresh, select_x_train.shape[1],
          score*100))

print('\n#############################################\n')
##############################################
# 3번
selection = SelectFromModel(model.best_estimator_, threshold = 0.011, prefit = True)
selection_x_train = selection.transform(x_train)
selection_x_test = selection.transform(x_test)

grid_model = GridSearchCV(model.best_estimator_, parameters)
grid_model.fit(selection_x_train, y_train)
score = grid_model.score(selection_x_test, y_test)
print('Thresh = %.3f, n = %d, R2: %.2f%%' %(0.011, selection_x_train.shape[1], score*100))