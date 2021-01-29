# RandomForest 사용
# pipeline 엮어서 25번 돌리기!

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target


# 2. 모델
parameters = [
    {'forest__n_estimators': [100, 200]},
    {'forest__max_depth': [6, 8, 10, 12]},
    {'forest__min_samples_leaf': [3, 5, 6, 10]},
    {'forest__min_samples_split': [2, 3, 5, 10]},
    {'forest__n_jobs': [-1, 2, 4]}
]
