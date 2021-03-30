import numpy as np
import tensorflow as tf

#1. data
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. model making
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim = 1, activation= "linear"))
model.add(Dense(3, activation = "linear"))
model.add(Dense(4))
model.add(Dense(1))

#3. complie and traning
from tensorflow.keras.optimizers import Adam, SGD
optimizer = Adam(learning_rate=0.1)
model.compile(loss = "mse", optimizer = "adam")
# optimizer = SGD(learning_rate=0.1)

model.fit(x, y, epochs = 100, batch_size = 1)   # epochs = 반복횟수, batch_size = 한번에 처리할 데이터 수

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size = 1)
print("loss: ", loss)

x_pred = np.array([4])
result = model.predict(x_pred)
# result = model.predict([4]) # 매개변수는 x값임
print("result: ", result)

