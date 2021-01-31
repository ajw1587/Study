# 실습
# RandomSearch와 Pipeline을 엮어라!
# 모델은 RandomForest

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier


# 1. 데이터
dataset = load_wine()
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
# pipe = make_pipe(MinMaxScaler(), RandomForestClassifier())
# pipe = Pipeline([('scaler', StandardScaler()), ('forest', RandomForestClassifier())])

# model = RandomizedSearchCV(pipe, parameters, cv = kfold)

# model.fit(x_train, y_train)


model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)
# pipe = Pipeline([('scaler', StandardScaler()), ('forest', model)])
pipe = make_pipeline(StandardScaler(), model)
pipe.fit(x_train, y_train)


result = model.score(x_test, y_test)
print(result)

# MinMaxScaler, RandomizedSearchCV
# 0.9722222222222222

# MinMaxScaler, GridSearchCV
# 1.0

# StandardScaler, RandomizedSearchCV
# 1.0

# StandardScaler, GridSearchCV
# 1.0