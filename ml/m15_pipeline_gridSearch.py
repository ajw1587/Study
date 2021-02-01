import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # 이름만 회귀이고 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)


# parameters = [
#     {'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['linear']},                                # 4번
#     {'svc__C': [1, 10, 100], 'svc__kernel': ['rbf'], 'svc__gamma': [0.001, 0.0001]},               # 6번
#     {'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['sigmoid'], 'svc__gamma': [0.001, 0.0001]}      # 8번
# ]       # SVC Model에 들어가있는 parameter 설정

parameters = [
    {'mal__C': [1, 10, 100, 1000], 'mal__kernel': ['linear']},                                # 4번
    {'mal__C': [1, 10, 100], 'mal__kernel': ['rbf'], 'mal__gamma': [0.001, 0.0001]},               # 6번
    {'mal__C': [1, 10, 100, 1000], 'mal__kernel': ['sigmoid'], 'mal__gamma': [0.001, 0.0001]}      # 8번
]       # SVC Model에 들어가있는 parameter 설정


# 2. 모델, pipeline: 모델 + 전처리
pipe = Pipeline([('scaler', MinMaxScaler()), ('mal', SVC())])
#                   ㄴ 아무 이름을 써도 상관X     ㄴ parameters 인자들의 이름과 맞춰줘야 한다.
# pipe = make_pipeline(MinMaxScaler(), SVC())
#                                       ㄴ 인자들의 이름이 없어 GridSearchCV에서 인지하지를 못한다.

model = GridSearchCV(pipe, parameters, cv = 5) # pipe를 써주면 cv(5번) 할때마다 전처리를 새롭게 해준다. 과적합 방지

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)

# MinMaxScaler() 의 결과 0.8333
# StandardScaler() 의 결과 0.8667