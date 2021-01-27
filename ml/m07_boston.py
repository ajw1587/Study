import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

dataset = load_boston()
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
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()

# 4. Fit, Evaluate
model.fit(x_train, y_train)

result1 = model.score(x_test, y_test)
print(result1)

y_pred = model.predict(x_test)
result2 = r2_score(y_test, y_pred)
print(result2)

# LinearRegression
# score:    0.7015599730509313
# r2_score: 0.7015599730509313

# KNeighborsRegressor
# score:    0.8091025066917995
# r2_score: 0.8091025066917995

# DecisionTreeRegressor
# score:    0.8345967060345953
# r2_score: 0.8345967060345953

# RandomForestRegressor
# score:    0.9200174606263768
# r2_score: 0.9200174606263768

# Tensorflow
# 0.9183033302819957