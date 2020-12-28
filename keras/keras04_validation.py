# 네이밍 룰

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential  # tensorflow 내부에 keras 내부에 models 내부에 Sequential
from tensorflow.keras.layers import Dense       # tensorflow 내부에 keras 내부에 layers 내부에 Dense

# 1. Data
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([1, 2, 3, 4, 5])

x_validation = np.array([6, 7, 8])  # 검증용 데이터
y_validation = np.array([6, 7, 8])

x_test = np.array([9, 10, 11])
y_test = np.array([9, 10, 11])

# 2. Making Model
model = Sequential()
model.add(Dense(5, input_dim = 1, activation = "relu"))
model.add(Dense(4))     # 뒤에 activation 붙이지 않으면 기본값인 'linear'적용
model.add(Dense(2))
model.add(Dense(1))

# 3. Compile and Traning
# model.compile(loss = "mse", optimizer = "adam", metrics = ["accuracy"])     # accuracy = acc -> 정확도 -> 특정한 경우 제외하고는 acc값은 0이다 ex) 개 or 고양이, 이진분리
# model.compile(loss = "mse", optimizer = "adam", metrics = ["mse"])
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"]) # mae: 평균절대오차

model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data=(x_validation, y_validation))

# 4. Evaluate and Predict       좋은 예측값을 얻기 위한 활동: 하이퍼 액티바이저
loss = model.evaluate(x_test, y_test, batch_size = 1)
print("loss: ", loss)

# result = model.predict([9])
result = model.predict(x_train)
print("result: ", result)