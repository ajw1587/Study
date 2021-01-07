# 행이 다른 모델에 대해서 공부
# 행은 맞춰줘야 한다.
# 밑의 에러가 뜬다.
# ValueError: Data cardinality is ambiguous:
#   x sizes: 6, 8
#   y sizes: 6, 8
# Please provide data which shares the same first dimension.

import numpy as np
from numpy import array

# 1. 데이터
x1 = array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9],
             [8, 9, 10], [9, 10, 11], [10, 11, 12]])        # [20, 30, 40], [30, 40, 50], [40, 50, 60]

x2 = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60], [50, 60, 70], [60, 70, 80], [70, 80, 90],
             [80, 90, 100], [90, 100, 110], [100, 110, 120], [2, 3, 4], [3, 4, 5], [4, 5, 6]])

y1 = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])      # 50, 60, 70
y2 = array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x1_predict = array([55, 65, 75])
x2_predict = array([65, 75, 85])

print("x1.shape: ", x1.shape)       # (10, 3)
print("x2.shape: ", x2.shape)       # (13, 3)
print("y1.shape: ", y1.shape)         # (10,)
print("y2.shape: ", y2.shape)         # (13,)
x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)

# 실습: ensemble 모델을 만드시오.
# 2. 데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size = 0.8, random_state = 70)
x1_train, x1_val, y1_train, y1_val = train_test_split(x1_train, y1_train, train_size = 0.8, random_state = 70)

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size = 0.8, random_state = 70)
x2_train, x2_val, y2_train, y2_val = train_test_split(x2_train, y2_train, train_size = 0.8, random_state = 70)

# 3. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM

input1 = Input(shape = (3,1), name = "input1")
lstm1 = LSTM(10, activation = "relu", name = "input1-1")(input1)
dense1 = Dense(200, activation = "relu", name = "input1-2")(lstm1)
dense1 = Dense(300, activation = "relu", name = "input1-3")(dense1)
dense1 = Dense(300, activation = "relu", name = "input1-4")(dense1)
dense1 = Dense(200, activation = "relu", name = "input1-5")(dense1)

input2 = Input(shape = (3,1), name = "input2")
lstm2 = LSTM(10, activation = "relu", name = "input2-1")(input2)
dense2 = Dense(200, activation = "relu", name = "input2-2")(lstm2)
dense2 = Dense(300, activation = "relu", name = "input2-3")(dense2)
dense2 = Dense(300, activation = "relu", name = "input2-4")(dense2)
dense2 = Dense(200, activation = "relu", name = "input2-5")(dense2)

# 모델 병0합
from tensorflow.keras.layers import concatenate
merge = concatenate([dense1, dense2])
middle = Dense(200, activation = "relu", name = "middle1")(merge)
middle = Dense(300, activation = "relu", name = "middle2")(middle)
middle = Dense(300, activation = "relu", name = "middle3")(middle)
middle = Dense(300, activation = "relu", name = "middle4")(middle)

middle1 = Dense(50, activation = "relu", name = "output1")(middle)
middle1 = Dense(30, activation = "relu", name = "output1-1")(middle1)
output1 = Dense(1, name = "output1-2")(middle1)

middle2 = Dense(50, activation = "relu", name = "output2")(middle)
middle2 = Dense(30, activation = "relu", name = "output2-1")(middle2)
output2 = Dense(1, name = "output2-2")(middle2)

model = Model(inputs = [input1, input2], outputs = [output1, output2])

# 4. Compile and Fit
model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit([x1_train, x2_train], [y1_train, y2_train], batch_size = 3, epochs = 200, validation_data = ([x1_val, x2_val], [y1_val, y2_val]))

# 5. Evaluate and Predict
loss, mae = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size = 3)
y1_test_predict, y2_test_predict = model.predict([x1_test, x2_test])

# # RMSE
# from sklearn.metrics import mean_squared_error
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))
# RMSE_1 = RMSE(y1_test, y_test_predict)
# print("1. RMSE: ", RMSE_1)

# # R2
# from sklearn.metrics import r2_score
# print("2. R2: ", r2_score(y1_test, y_test_predict))

# y_predict
x1_predict = x1_predict.reshape(1, 3, 1)
x2_predict = x2_predict.reshape(1, 3, 1)
y1_predict, y2_predict = model.predict([x1_predict, x2_predict])
print("y1_predict: \n", y1_predict)
print("y2_predict: \n", y2_predict)

# predict는 85의 근사치