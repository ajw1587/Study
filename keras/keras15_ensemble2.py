# 실습: 다:1(2:1) ensemble 만들기

import numpy as np

# 1. 데이터
x1 = np.array([range(100), range(301,401), range(1, 101)])       #(3,100)
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])

y1 = np.array([range(711, 811), range(1, 101), range(201, 301)]) #(3,100)
# y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
# y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = train_test_split(x1, x2, y1, train_size = 0.8, shuffle = False)
# x2_train, x2_test = train_test_split(x2, train_size = 0.8, shuffle = False)

# 2. 모델 구성
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

# 2-1. 모델 1
input1 = Input(shape = (3,))
dense1 = Dense(10, activation = "relu")(input1)
dense1 = Dense(5, activation = "relu")(dense1)
# output1 = Dense(3)(dense1)

# 2-2. 모델 2
input2 = Input(shape = (3,))
dense2 = Dense(10, activation = "relu")(input2)
dense2 = Dense(5, activation = "relu")(dense2)
dense2 = Dense(5, activation = "relu")(dense2)
dense2 = Dense(5, activation = "relu")(dense2)
# output2 = Dense(3)(dense1)

# 모델 병합 / concatenate: 연쇄시키다
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenatem Concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

# 모델 분기 1
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)
# 모델 분기 2
# output2 = Dense(15)(middle1)
# output2 = Dense(7)(output2)
# output2 = Dense(7)(output2)
# output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs = [input1, input2], outputs = output1)
model.summary()

# 훈련
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit([x1_train, x2_train], y1_train, epochs = 10, batch_size = 1, validation_split = 0.2, verbose = 1)

# 평가, 예측
loss, mae = model.evaluate([x1_test, x2_test], y1_test, batch_size = 1)
print("loss: ", loss)                           # loss 5개의 값이 나온다. [분기모델1+분기모델2의 mse합, output1의 mse, output2의 mse, output1의 metrics값, output2의 metrics값]
print("mae: ", mae)
print("metrics_name: ", model.metrics_names)    # loss 5개의 값이 나온다. [분기모델1+분기모델2의 mse합, output1의 mse, output2의 mse, output1의 metrics값, output2의 metrics값]
                                                # loss:  [8131.48291015625, 4593.9697265625, 3537.51318359375, 65.62861633300781, 54.035789489746094]
                                                # metrics_name:  ['loss', 'dense_11_loss', 'dense_15_loss', 'dense_11_mae', 'dense_15_mae']

y1_predict = model.predict([x1_test, x2_test])
print("=============================================")
print("y1_predict\n", y1_predict)
print("=============================================")  
# print("y2_predict\n", y2_predict)
# print("=============================================")

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE_1 = RMSE(y1_test, y1_predict)
# RMSE_2 = RMSE(y2_test, y2_predict)
print("1. RMSE: ", RMSE_1)
# print("2. RMSE: ", RMSE_2)
# print("RMSE: ", (RMSE_1+RMSE_2)/2)

# R2
from sklearn.metrics import r2_score
print("1. R2: ", r2_score(y1_test, y1_predict))
# print("2. R2: ", r2_score(y2_test, y2_predict))
# print("R2: ", (r2_score(y1_test, y1_predict)+r2_score(y2_test, y2_predict)))