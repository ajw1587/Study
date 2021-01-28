# RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

# 2. Model
parameters = [
    {'n_estimators': [100, 200]},
    {'max_depth': [6, 8, 10, 12]},
    {'min_samples_leaf': [3, 5, 6, 10]},
    {'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1, 2, 4]}
]
kfold = KFold(n_splits = 5, shuffle = True)

model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)

# 3. Fit
model.fit(x_train, y_train)

# 4. 예측, 평가
print('Best Parameters: ', model.best_estimator_)
print('Accuracy: ', model.score(x_test, y_test))

# Best Parameters:  RandomForestClassifier(n_jobs=-1)
# Accuracy:  0.9722222222222222