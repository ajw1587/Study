# RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
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
model = RandomizedSearchCV(XGBClassifier(n_jobs = -1), parameters, cv = kfold)

# 3. Fit
model.fit(x_train, y_train)

# 4. 예측, 평가
print('Best Parameters: ', model.best_estimator_)
print('Accuracy: ', model.score(x_test, y_test))