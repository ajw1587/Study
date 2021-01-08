# EarlyStopping: 과적합 되기 전에 멈추는것
# 실습
# 드랍아웃 적용

import numpy as np
from sklearn.datasets import load_boston

# 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)      # (506, 13)
print(y.shape)      # (506,)
print("=========================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))
print(dataset.feature_names)
# print(dataset.DESCR)


# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 66)
print(x_train.shape)
print(y_train.shape)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 모델 구성
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

input1 = Input(shape = (13,))
dense1 = Dense(50, activation = "relu")(input1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)       # 200개중 0.8개만 사용하겠다
dense1 = Dense(400, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)       # 400개중 0.8개만 사용하겠다
dense1 = Dense(500, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)       # 500개중 0.8개만 사용하겠다
dense1 = Dense(400, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)       # 400개중 0.8개만 사용하겠다
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)       # 200개중 0.8개만 사용하겠다
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(50, activation = "relu")(dense1)
output1 = Dense(1, activation = "relu")(dense1)
model = Model(inputs = input1, outputs = output1)

# compile and fit
model.compile(loss = "mse", optimizer = "adam", metrics = "mae")

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 10, mode = "auto")

model.fit(x_train, y_train, batch_size = 6, epochs = 2000, validation_data = (x_val, y_val), callbacks = [early_stopping])

# 평가 및 예측
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("mae: ", mae)

y_predict = model.predict(x_test)

print("RMSE: ", RMSE(y_test, y_predict))
print("R2_SCORE: ", r2_score(y_test, y_predict))

# Dropout 적용 전
'''
loss:  5.448845386505127
mae:  1.6925382614135742
RMSE:  2.3342761908836756
R2_SCORE:  0.9348091188313188
'''

# Dropout 적용 후
'''
loss:  12.099385261535645
mae:  2.919724464416504
RMSE:  3.478416994818806
R2_SCORE:  0.855240971709167
'''