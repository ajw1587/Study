# 싸이킷런 데이터 셋
# LSTM으로 모델링
# Dense와 성능 비교
# 다중분류

import numpy as np

# 데이터
from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)         # (150, 4)
print(np.max(x), np.min(x))     # (150,)
print(np.max(x[0]), np.min(x[0]))

# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2, random_state = 66)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
print(y_train[:20])

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape = (4, 1))
lstm1 = LSTM(10, activation = "relu")(input1)
dense1 = Dense(100, activation = "relu")(lstm1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(10, activation = "relu")(dense1)
output1 = Dense(3,  activation = "softmax")(dense1)
model = Model(inputs = input1, outputs = output1)


# Compile and Fit and Early_Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 30, mode = "auto")
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["acc"])
model.fit(x_train, y_train, epochs = 200, batch_size = 10, validation_data = (x_val, y_val), callbacks = early_stopping)


# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
y_test_predict = model.predict(x_test)
print("y_test_predict: ", y_test_predict.shape)
print("y_test_predict: \n", y_test_predict)
print("y_test: \n", y_test)

loss = model.evaluate(x_test, y_test, batch_size = 10)
rmse = RMSE(y_test, y_test_predict)
R2 = r2_score(y_test, y_test_predict)

print("loss: ", loss)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2)

result = model.predict(x_test)
print("result: \n", result)

# loss:  [0.05408873409032822, 0.12230633944272995]
# RMSE:  0.23256982196764472
# R2_SCORE:  0.91832239952843

# loss:  [0.04638289660215378, 0.12418417632579803]
# RMSE:  0.21536692486890077
# R2_SCORE:  0.9299586894383589
