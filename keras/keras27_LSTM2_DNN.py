# keras23_LSTM3_scale 을 DNN으로 코딩
# 결과치 비교

import numpy as np
# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x_pred = np.array([50, 60, 70])

print("x.shape: ", x.shape)     # (13, 3) LSTM3_scaler에서 13개의 모델을 가지기 떄문에 동일시 해주려고 shape를 안바꿨다.
print("y.shape: ", y.shape)     # (13,)
print(y)

# 데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 70)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 70)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
print("x_train: ", x_train)
x_test = scaler.transform(x_test)
print("x_test: ", x_test)
x_val = scaler.transform(x_val)
print("x_val: ", x_val)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation = "relu", input_shape = (3,)))
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 30, mode = "auto")

model.compile(loss = "mse", optimizer = "adam")
model.fit(x_train, y_train, epochs = 150, batch_size = 1, validation_data = (x_val, y_val), callbacks = early_stopping)

# 4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

x_pred = x_pred.reshape(1,3)
x_pred = scaler.transform(x_pred)
result = model.predict(x_pred)
print("result: ", result)

# LSTM
# loss:  0.4433615803718567
# result:  [[80.322945]]

# loss:  0.5716427564620972
# result:  [[80.266685]]

# loss:  0.9403667449951172
# result:  [[78.230125]]

# loss:  0.3914860188961029
# result:  [[81.646996]]

# ====================================================

# DNN
# loss:  0.3382943868637085
# result:  [[82.46935]]

# loss:  3.9503591060638428
# result:  [[84.70415]]

# loss:  0.0006672668387182057
# result:  [[80.102036]]

# loss:  0.0014844370307400823
# result:  [[80.037254]]