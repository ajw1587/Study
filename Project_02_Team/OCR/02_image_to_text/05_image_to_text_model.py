# metrics가 다를 경우

import numpy as np

# 1. 데이터
x1 = np.array([range(100), range(301,401), range(1, 101)])       #(3,100)
y1 = np.array([range(711, 811), range(1, 101), range(201, 301)]) #(3,100)

x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size = 0.8, shuffle = False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size = 0.8, shuffle = False)

# 2. 모델 구성
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input

# 2-1. 모델 1
input1 = Input(shape = (3,), name = "model1")
dense1 = Dense(10, activation = "relu", name = "model1-1")(input1)
dense1 = Dense(5, activation = "relu", name = "model1-2")(dense1)
# output1 = Dense(3)(dense1)

# 2-2. 모델 2
input2 = Input(shape = (3,), name = "model2")
dense2 = Dense(5, activation = "relu", name = "model2-1")(input2)
dense2 = Dense(5, activation = "relu", name = "model2-2")(dense2)
dense2 = Dense(5, activation = "relu", name = "model2-3")(dense2)
# output2 = Dense(3)(dense1)

# 2-3. 모델 3
input3 = Input(shape = (3,), name = "model3")
dense3 = Dense(5, activation = "relu", name = "model3-1")(input3)
dense3 = Dense(5, activation = "relu", name = "model3-2")(dense3)
dense3 = Dense(5, activation = "relu", name = "model3-3")(dense3)

# 모델 병합 / concatenate: 연쇄시키다
from tensorflow.keras.layers import Concatenate, concatenate
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenatem Concatenate
merge1 = concatenate([dense1, dense2, dense3])
middle1 = Dense(3, name = "merge-1")(merge1)
middle1 = Dense(5, name = "merge-2")(middle1)
middle1 = Dense(5, name = "merge-3")(middle1)

# 모델 분기 1
output1 = Dense(3, name = "output1-1")(middle1)
output1 = Dense(7, name = "output1-2")(output1)
output1 = Dense(3, name = "output1-3")(output1)
# 모델 분기 2
output2 = Dense(1, name = "output2-1")(middle1)
output2 = Dense(5, name = "output2-2")(output2)
output2 = Dense(3, name = "output2-3")(output2)

# 모델 선언
model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()

# compile
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae", "mse"])   # metrics에 2개의 값 사용 가능
model.fit([x1_train, x2_train, x2_train], [y1_train, y2_train], epochs = 10, batch_size = 1, validation_split = 0.2, verbose = 1)

# 평가, 예측
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size = 1)
print("loss: ", loss)                           # loss 7개의 값이 나온다.

print("metrics_name: ", model.metrics_names)    # loss 7개의 값이 나온다. [분기모델1+분기모델2의 mse합, output1의 mse, output2의 mse, output1의 metrics값, output2의 metrics값]
                                                # loss:  [4844.7060546875, 1856.223388671875, 2988.483154296875, 39.791404724121094, 1856.223388671875, 52.21002960205078, 2988.483154296875]
                                                # metrics_name: ['loss', 'output1-3_loss', 'output2-3_loss', 'output1-3_mae', 'output1-3_mse', 'output2-3_mae', 'output2-3_mse']

y1_predict, y2_predict = model.predict([x1_test, x2_test])
# print("=============================================")
# print("y1_predict\n", y1_predict)
# print("=============================================")
# print("y2_predict\n", y2_predict)
# print("=============================================")

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
RMSE_1 = RMSE(y1_test, y1_predict)
RMSE_2 = RMSE(y2_test, y2_predict)
print("1. RMSE: ", RMSE_1)
print("2. RMSE: ", RMSE_2)
print("RMSE: ", (RMSE_1+RMSE_2)/2)

# R2
from sklearn.metrics import r2_score
print("1. R2: ", r2_score(y1_test, y1_predict))
print("2. R2: ", r2_score(y2_test, y2_predict))
print("R2: ", (r2_score(y1_test, y1_predict)+r2_score(y2_test, y2_predict)))