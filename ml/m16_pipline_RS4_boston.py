# 실습
# RandomSearch와 Pipeline을 엮어라!
# 모델은 RandomForest

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor


# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

# 2. 모델
parameters = [
    {'forest__n_estimators': [100, 200]},
    {'forest__max_depth': [6, 8, 10, 12]},
    {'forest__min_samples_leaf': [3, 5, 6, 10]},
    {'forest__min_samples_split': [2, 3, 5, 10]},
    {'forest__n_jobs': [-1, 2, 4]}
]
kfold = KFold(n_splits = 5, shuffle = True)
# pipe = make_pipe(MinMaxScaler(), RandomForestRegressor())
pipe = Pipeline([('scaler', StandardScaler()), ('forest', RandomForestRegressor())])

model = RandomizedSearchCV(pipe, parameters, cv = kfold)

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)

# MinMaxScaler, RandomizedSearchCV
# 0.899535752732143

# MinMaxScaler, GridSearchCV
# 0.8960189569689045

# StandardScaler, RandomizedSearchCV
# 0.8966589040451534

# StandardScaler, GridSearchCV
# 0.8956407369523586