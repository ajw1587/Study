# 실습
# RandomSearch와 Pipeline을 엮어라!
# 모델은 RandomForest

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, train_test_split

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)


# 2. 모델
parameters = [
    {'forest__n_estimators': [100, 200]},
    {'forest__max_depth': [6, 8, 10, 12]},
    {'forest__min_samples_leaf': [3, 5, 6, 10]},
    {'forest__min_samples_split': [2, 3, 5, 10]},
    {'forest__n_jobs': [-1, 2, 4]}
]
kfold = KFold(n_splits = 5, shuffle = True)

# pipe = make_pipeline(MinMaxScaler(), RandomizedSearchCV())
pipe = Pipeline([('scaler', StandardScaler()), ('forest', RandomForestClassifier())])

model = RandomizedSearchCV(pipe, parameters, cv = kfold)
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)

# MinMaxScaler, RandomizedSearchCV
# 0.8333333333333334

# MinMaxScaler, GridSearchCV
# 0.8666666666666667

# StandardScaler, RandomizedSearchCV
# 0.8666666666666667

# StandardScaler, GridSearchCV
# 0.8666666666666667