import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
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

dataset = load_wine()
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
         LogisticRegression(),          # 2진분류라 warning 발생
         DecisionTreeClassifier(),
         RandomForestClassifier()]          
for i in range(6):
    scores = cross_val_score(model[i], x_train, y_train, cv=kfold)      # model과 Data를 엮어주는 역할
    print(model[i], 'scores: ', scores)                                        # model이 5번 fit해서 값이 5개이다.

'''
# 4. Compile and Train
model.fit(x_train, y_train)

result1 = model.score(x_test, y_test)
print(result1)

y_predict = model.predict(x_test)
result2 = accuracy_score(y_test, y_predict)
print(result2)
'''
# LinearSVC() scores:  [0.89655172 0.89655172 0.60714286 0.85714286 0.75      ]
# SVC() scores:  [0.68965517 0.75862069 0.46428571 0.67857143 0.78571429]
# KNeighborsClassifier() scores:  [0.75862069 0.62068966 0.71428571 0.60714286 0.67857143]
# LogisticRegression() scores:  [0.93103448 1.         0.92857143 0.85714286 0.96428571]
# DecisionTreeClassifier() scores:  [0.96551724 0.93103448 0.92857143 0.92857143 0.89285714]
# RandomForestClassifier() scores:  [0.93103448 0.93103448 1.         1.         0.96428571]