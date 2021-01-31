# RandomForest 사용
# pipeline 엮어서 25번 돌리기!

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
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

model1 = RandomizedSearchCV(RandomForestClassifier(), parameters, cv = kfold)
model2 = Pipeline([('scaler', StandardScaler()), ('forest', model1)])
model3 = RandomizedSearchCV(model2, parameters, cv = kfold)

# 3. Fit
model3.fit(x, y)

# 4. Score
acc_result = model3.score(x, y)
print(acc_result)