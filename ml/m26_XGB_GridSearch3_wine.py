# RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

# 2. Model
kfold = KFold(n_splits = 5, shuffle = True)
parameters = [
    {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 0.001, 0.01], 'max_depth': [4, 5, 6]},
    {'n_estimators': [90, 100, 110], 'learning_rate': [0.1, 0.001, 0.01], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.9, 1]},
    {'n_estimators': [90, 110], 'learning_rate': [0.1, 0.001, 0.5], 'max_depth': [4, 5, 6], 'colsample_bytree':[0.6, 0.7, 0.9]}
]
model = GridSearchCV(XGBClassifier(n_jobs = -1), parameters, cv = kfold)

# 3. Fit
model.fit(x_train, y_train)

# 4. 예측, 평가
print('Best Parameters: ', model.best_estimator_)
print('Accuracy: ', model.score(x_test, y_test))

# Best Parameters:  XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.6, gamma=0, gpu_id=-1,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.1, max_delta_step=0, max_depth=4,
#               min_child_weight=1, missing=nan, monotone_constraints='()',
#               n_estimators=90, n_jobs=-1, num_parallel_tree=1,
#               objective='multi:softprob', random_state=0, reg_alpha=0,
#               reg_lambda=1, scale_pos_weight=None, subsample=1,
#               tree_method='exact', validate_parameters=1, verbosity=None)
# Accuracy:  1.0