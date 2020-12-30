# 다:다 mlp
# keras10_mlp3.py를 함수형으로 바꾸시오

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

# 데이터
x = np.array([range(100), range(301,401), range(1, 101)])       #(3,100)
y = np.array([range(711, 811), range(1, 101), range(201, 301)]) #(3,100)

x = np.transpose(x)                                             #(100,3)
y = np.transpose(y)                                             #(100,3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

# 모델 구성
input1 = Input(shape = 3)
dense1 = Dense(10)(input1)
dense1 = Dense(5)(dense1)
output1 = Dense(3)(dense1)
model = Model(inputs = input1, outputs = output1)

# 컴파일
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train ,y_train, epochs = 100, batch_size = 1, validation_split=0.2)

# 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("mae: ", mae)

y_predict = model.predict(x_test)
print("y_predict: ", y_predict)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print("RMSE: ", RMSE(y_test, y_predict))
print("R2: ", r2)