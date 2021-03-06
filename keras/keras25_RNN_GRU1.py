# 1. 데이터
import numpy as np

x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
y = np.array([4, 5, 6, 7])

print("x.shape: ", x.shape)     # (4, 3)
print("y.shape: ", y.shape)     # (4,)

x = x.reshape(4, 3, 1)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
model = Sequential()
model.add(GRU(10, activation = "relu", input_shape = (3, 1)))      # LSTM의 Activation Default값은 tanh
# Ramda값: 3*(n*(n+m)+n)
# -> n = 10, m = 1, 3: Gate 개수, 1: bias
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

model.summary()

'''
# 3. 컴파일, 훈련
model.compile(loss = "mse", optimizer = "adam")
model.fit(x, y, epochs = 100, batch_size = 1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss: ", loss)

x_pred = np.array([5, 6, 7])    # (3,) -> (1, 3, 1)
x_pred = x_pred.reshape(1, 3, 1)

result = model.predict(x_pred)
print("result: ", result)
'''