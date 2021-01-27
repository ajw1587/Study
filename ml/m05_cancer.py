import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 이름만 회구이고 분류모델이다.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 분류에서는 score가 accuracy_score로 잡힌다.
# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

# 2. 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 3. 모델
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()

# 4. Fit, Evaluate
model.fit(x_train, y_train)

result1 = model.score(x_test, y_test)
print(result1)

y_pred = model.predict(x_test)
result2 = accuracy_score(y_test, y_pred)
print(result2)

# LinearSVC
# score:   0.9824561403508771
# Accuray: 0.9824561403508771

# SVC
# score:   0.9912280701754386
# Accuray: 0.9912280701754386
# SVC > LinearSVC

# KNeighborsClassifier
# score:   0.956140350877193
# Accuray: 0.956140350877193

# LogisticRegression
# score:   0.956140350877193
# Accuray: 0.956140350877193

# DecisionTreeClassifier
# score:   0.8947368421052632
# Accuray: 0.8947368421052632

# RandomForestClassifier
# score:   0.956140350877193
# Accuray: 0.956140350877193
# RandomForestClassifier > DecisionTreeClassifier

# Tensorflow
# 0.9912280440330505