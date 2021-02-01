# 모델: RandomForestClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)
kfold = KFold(n_splits = 5, shuffle = True)

# 2. Model
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
# y_pred = model.predict(x_test)
# print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Accuracy: ', model.score(x_test, y_test))


##################################################################################

# n_jobs
# - 사용할 코어 수