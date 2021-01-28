# RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
dataset = load_boston()
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

model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv = kfold)

# 3. Fit
model.fit(x_train, y_train)

# 4. 예측, 평가
print('Best Parameters: ', model.best_estimator_)
print('R2_SCORE: ', model.score(x_test, y_test))

# Best Parameters:  RandomForestRegressor(n_estimators=200)
# R2_SCORE:  0.8983753212421203