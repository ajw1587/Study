import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = load_wine()
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
# score:   1.0
# Accuray: 1.0

# SVC
# score:   1.0
# Accuray: 1.0
# SVC = LinearSVC

# KNeighborsClassifier
# score:   0.8888888888888888
# Accuray: 0.8888888888888888

# LogisticRegression
# score:   0.9722222222222222
# Accuray: 0.9722222222222222

# DecisionTreeClassifier
# score:   0.9166666666666666
# Accuray: 0.9166666666666666

# RandomForestClassifier
# score:   0.9722222222222222
# Accuray: 0.9722222222222222
# RandomForestClassifier > DecisionTreeClassifier

# Tensorflow
# 1.0