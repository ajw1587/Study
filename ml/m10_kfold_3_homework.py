# 실습 또는 과제
# train, test 나눈후 train만 validation 하지 말고,
# kfold 한 후에 train, test 나누기 (train_test_split 사용X)

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
import warnings
warnings.filterwarnings('ignore')
# 분류에서는 score가 accuracy_score로 잡힌다.
# 1. 데이터
# x, y = load_iris(return_X_y= True)        x, y를 나눠주는 또다른 방법

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(y)            # 0, 1, 2

# 2. k-fold 전처리
kfold = KFold(n_splits = 5, shuffle = True)     # n_splits: 몇도막으로 나누겠다.

# 3. Model
model = [LinearSVC(),
         SVC(),
         KNeighborsClassifier(),
         LogisticRegression(),
         DecisionTreeClassifier(),
         RandomForestClassifier()]

for i in range(6):
    score = []
    for train_index, test_index in kfold.split(x):
        # print('train_index: ', train_index)
        # print('test_index: ', test_index)

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        md = model[i]
        md.fit(x_train, y_train)
        y_pred = md.predict(x_test)
        result = accuracy_score(y_test, y_pred)
        score.append(result)
    score = list(map(float, score))
    print(model[i], '의 Accuracy: ', np.round(score, 4))


'''
# 3. 모델
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

scores = cross_val_score(model, x_train, y_train, cv=kfold)      # model과 Data를 엮어주는 역할
print('scores: ', scores)                            # model이 5번 fit해서 값이 5개이다.


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