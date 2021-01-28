import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 이름만 회귀이고 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 분류에서는 score가 accuracy_score로 잡힌다.
# 1. 데이터
# x, y = load_iris(return_X_y= True)        x, y를 나눠주는 또다른 방법

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(y)            # 0, 1, 2

# 2. k-fold 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 77, shuffle = True, train_size = 0.8)
kfold = KFold(n_splits = 5, shuffle = True)     # n_splits: 몇도막으로 나누겠다.

# 3. 모델
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

scores = cross_val_score(model, x_train, y_train, cv=kfold)      # model과 Data를 엮어주는 역할
print('scores: ', scores)                            # model이 5번 fit해서 값이 5개이다.

'''
# 4. Compile and Train
model.fit(x_train, y_train)

result1 = model.score(x_test, y_test)
print(result1)

y_predict = model.predict(x_test)
result2 = accuracy_score(y_test, y_predict)
print(result2)
'''
# LinearSVC
# score:   0.9
# Accuray: 0.9

# SVC
# score:   1.0
# Accuray: 1.0
# SVC > LinearSVC

# KNeighborsClassifier
# score:   1.0
# Accuray: 1.0

# LogisticRegression
# score:   0.9666666666666667
# Accuray: 0.9666666666666667

# DecisionTreeClassifier
# score:   0.9333333333333333
# Accuray: 0.9333333333333333

# RandomForestClassifier
# score:   0.9666666666666667
# Accuray: 0.9666666666666667
# RandomForestClassifier > DecisionTreeClassifier

# Tensorflow
# 0.9666666388511658