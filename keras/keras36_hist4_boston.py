# hist를 이용하여 그래프를 그리시오.
# loss, val_loss, acc, val_acc

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
print(dataset.DESCR)

# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 7)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 7)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 모델 구성
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

input1 = Input(shape = (13,))
dense1 = Dense(50, activation = "relu")(input1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
# dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(50, activation = "relu")(dense1)
output1 = Dense(1, activation = "relu")(dense1)
model = Model(inputs = input1, outputs = output1)

# compile and fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 15, mode = 'auro')
model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
hist = model.fit(x_train, y_train, batch_size = 6, epochs = 500, validation_data = (x_val, y_val), callbacks = es)

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


# Graph
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])
plt.title('loss & mae')
plt.ylabel('loss, mae')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train mae', 'val mae'])
plt.show()