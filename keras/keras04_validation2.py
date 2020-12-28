from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

# 1. Data
x_train = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = array([11, 12, 13, 14, 15])
y_test = array([11, 12, 13, 14, 15])
x_pred = array([16, 17, 18])

# 2. model
model = Sequential()
model.add(Dense(10, input_dim = 1, activation = "relu"))
model.add(Dense(5))
model.add(Dense(1))

# 3. compile and traning
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.2) # validation_split -> train 데이터에서 20%를 검증용 데이터로 사용

# 4. 예측 및 평가
results = model.evaluate(x_test, y_test, batch_size = 1)
print("results: ", results)

y_pred = model.predict(x_pred)
print("y_pred: ", "\n", y_pred)