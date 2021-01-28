# 모델: RandomForestClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)
kfold = KFold(n_splits = 5, shuffle = True)

# 2. Model
parameters = [
    {'n_estimators': [100, 200]},
    {'max_depth': [6, 8, 10, 12]},
    {'min_samples_leaf': [3, 5, 6, 10], 'n_estimators': [100, 200], 'max_depth': [6, 8, 10, 12],\
        'min_samples_split': [2, 3, 5, 10], 'n_jobs': [-1, 2, 4]},
    {'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1, 2, 4]}
]
model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold)

# 3. Fit
model.fit(x_train, y_train)

# 4. 예측, 평가
print('Best Parameters: ', model.best_estimator_)
# y_pred = model.predict(x_test)
# print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Accuracy: ', model.score(x_test, y_test))

# Best Parameters:  RandomForestClassifier(max_depth=6, min_samples_leaf=10, min_samples_split=5, n_jobs=4)
# Accuracy:  0.8666666666666667