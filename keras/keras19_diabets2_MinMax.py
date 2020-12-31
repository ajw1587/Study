import tensorflow as tf
import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data        #(442,10)
y = dataset.target      #(442,)

print(x[:5])
print(y[:10])
print(x.shape)
print(y.shape)

print(np.max(x), np.min(x))
print(np.max(x[0]), np.min(x[0]))
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 MinMax
x = x/np.max(x)

# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 77)

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape = (10,))
dense1 = Dense(30, activation = "relu")(input1)
dense1 = Dense(30, activation = "relu")(dense1)
dense1 = Dense(20, activation = "relu")(dense1)
dense1 = Dense(10, activation = "relu")(dense1)
dense1 = Dense(5, activation = "relu")(dense1)
output1 = Dense(1, activation = "relu")(dense1)
model = Model(inputs = input1, outputs = output1)

# Compile and fit
model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(x_train, y_train, epochs = 200, batch_size = 6, validation_split = 0.2)

# 평가 및 예측
from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

loss, mae = model.evaluate(x_test, y_test, batch_size = 6)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("mae: ", mae)
print("RMSE: ", RMSE(y_test, y_predict))
print("R2_SCORE: ", r2_score(y_test, y_predict))

# 일반
# loss:  3345.530517578125
# mae:  47.99224853515625
# RMSE:  57.8405595780664
# R2_SCORE:  0.48451317300728847


# x = x/(np.max(x))
# loss:  3579.1552734375
# mae:  49.40424346923828
# RMSE:  59.82604087926021
# R2_SCORE:  0.4328804440523756