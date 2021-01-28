import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 이름만 회귀이고 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# 분류에서는 score가 accuracy_score로 잡힌다.
# 1. 데이터
# x, y = load_iris(return_X_y= True)        x, y를 나눠주는 또다른 방법

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
# print(y)            # 0, 1, 2

# 2. k-fold 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 77, shuffle = True, train_size = 0.8)
kfold = KFold(n_splits = 5, shuffle = True)     # n_splits: 몇도막으로 나누겠다.

# 3. 모델
model = [LinearSVC(),
         SVC(),
         KNeighborsClassifier(),
         LogisticRegression(),
         DecisionTreeClassifier(),
         RandomForestClassifier()]          
for i in range(6):
    scores = cross_val_score(model[i], x_train, y_train, cv=kfold)      # model과 Data를 엮어주는 역할
    print(model[i], 'Accuracy: ', scores)                                 # model이 5번 fit해서 값이 5개이다.

'''
# 4. Compile and Train
model.fit(x_train, y_train)

result1 = model.score(x_test, y_test)
print(result1)

y_predict = model.predict(x_test)
result2 = accuracy_score(y_test, y_predict)
print(result2)
'''
# LinearSVC() scores:  [0.9010989  0.86813187 0.94505495 0.86813187 0.85714286]
# SVC() scores:  [0.89010989 0.91208791 0.92307692 0.9010989  0.92307692]
# KNeighborsClassifier() scores:  [0.93406593 0.91208791 0.92307692 0.93406593 0.91208791]
# LogisticRegression() scores:  [0.95604396 0.93406593 0.94505495 0.96703297 0.92307692]
# DecisionTreeClassifier() scores:  [0.94505495 0.91208791 0.91208791 0.91208791 0.9010989 ]
# RandomForestClassifier() scores:  [0.96703297 0.94505495 0.97802198 0.97802198 0.93406593]