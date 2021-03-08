# 표준편차를 이용한 전처리

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

# 데이터 전처리 (StandardScaler): 유독 튀는 값이 있을때 사용
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()                                # 디폴트: 균등분포
# scaler = QuantileTransformer(output_distribution = 'normal')  # 정규분호
# scaler = MaxAbsScaler()
scaler = PowerTransformer(method = 'yeo-johnson')
# scaler = Powertransformer(method = 'box-cox')

scaler.fit(x)
x = scaler.transform(x)

# 각 Column마다 최소, 최대값을 몰라도 MinMaxScaler를 사용하면 열(Column)마다 자동으로 0 ~ 1 사이로 만들어준다.
print(np.max(x), np.min(x))         # 9.933930601860268  -3.9071933049810337
print(np.max(x[0]))


# 데이터 분할
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 66)

# 모델 구성
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

input1 = Input(shape = (13,))
dense1 = Dense(50, activation = "relu")(input1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(400, activation = "relu")(dense1)
dense1 = Dense(500, activation = "relu")(dense1)
dense1 = Dense(400, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(50, activation = "relu")(dense1)
output1 = Dense(1, activation = "relu")(dense1)
model = Model(inputs = input1, outputs = output1)

# compile and fit
model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(x_train, y_train, batch_size = 6, epochs = 200, validation_split = 0.2)

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


# x = x/711.
# loss:  13.544418334960938
# mae:  2.8383567333221436
# RMSE:  3.6802741288102787
# R2_SCORE:  0.8379523609056285


# scaler = MinMaxScaler()
# loss:  14.707382202148438
# mae:  2.3291330337524414
# RMSE:  3.835020550618675
# R2_SCORE:  0.8240384569725716

# scaler = StandardScaler()
# loss:  16.221664428710938
# mae:  2.315497875213623
# RMSE:  4.027612626515545
# R2_SCORE:  0.8059213520473318