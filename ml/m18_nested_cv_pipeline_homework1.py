# RandomForest 사용
# pipeline 엮어서 25번 돌리기!

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target


# 2. 모델
parameters = [
    {'forest__n_estimators': [100]},              # , 200
    {'forest__max_depth': [6]},                   # , 8, 10, 12
    {'forest__min_samples_leaf': [3]},            # , 5, 6, 10
    {'forest__min_samples_split': [2]},           # , 3, 5, 10
    {'forest__n_jobs': [-1]}                      # , 2, 4
]
kfold = KFold(n_splits = 5, shuffle = True)

model1 = Pipeline([('scaler', StandardScaler()), ('forest', RandomForestClassifier())])
model2 = RandomizedSearchCV(model1, parameters, cv = kfold)
score = cross_val_score(model2, x, y, cv = kfold)
# model2에서 5번, score에서 5번 = 총 25번

# 3. Score
print('중첩 교차 검증: ', score)

# 중첩 교차 검증:  [0.97222222 0.97222222 1.         1.         0.94285714]