import numpy as np
# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x_pred = np.array([50, 60, 70])

# 코딩하시오!!! LSTM
# 원하는 답은 80
print("x.shape: ", x.shape)     # (13, 3)
print("y.shape: ", y.shape)     # (13,)
x = x.reshape(13, 3, 1)
print(x)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation = "relu", input_shape = (3, 1)))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

# Compile and Fit
model.compile(loss = "mse", optimizer = "adam")
model.fit(x, y, epochs = 150, batch_size = 1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss: ", loss)

x_pred = x_pred.reshape(1, 3, 1)        # (3,) -> (1, 3, 1)
print(x_pred)

result = model.predict(x_pred)
print("result: ", result)

# loss:  0.4433615803718567
# result:  [[80.322945]]

# loss:  0.5716427564620972
# result:  [[80.266685]]