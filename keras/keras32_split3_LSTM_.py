# 과제 및 실습  LSTM
# EarlyStopping, 전처리 등등 지금까지 배운 내용 다 넣기
# 데이터는 1~ 100 / x_size: 5
#     x               y
# 1,2,3,4,5           6
# ...
# 95,96,97,98,99    100

# predict 만들것
# 96,97,98,99,100 -> 101
# ...
# 100,101,102,103,104 -> 105
# 예상 predict는 (101, 102, 103, 104, 105)

# 1. 데이터
import numpy as np

sample_data = np.array(range(1,101))

def data_split(data, size):
    arr = []
    for i in range(len(data) - size + 1):
        subset = data[i : (i+size)]
        arr.append(subset)
    return np.array(arr)

dataset = data_split(sample_data, 6)
# print("dataset: \n", dataset)

x = dataset[:, :5]
y = dataset[:, 5:]
print("x.shape: ", x.shape)     # (95,5)
print("y.shape: ", y.shape)     # (95,1)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 70)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 70)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("x_val: ", x_val.shape)

print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
print("y_val: ", y_val.shape)


# Predict 데이터
sample_data2 = np.array(range(96, 106)) # 96 ~ 105
dataset2 = data_split(sample_data2, 6)
print("dataset2: \n", dataset2)

x_predict = dataset2[:, :5]
print(x_predict)

x_predict = scaler.transform(x_predict)
print(x_predict)

x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
print(x_predict)
print("x_predict.shape(reshape): ", x_predict.shape)

y_predict = dataset2[:, 5:]
print("x_predict.shape: ", x_predict.shape)     # (5, 5)
print("y_predict.shape: ", y_predict.shape)     # (5, 1)


# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape = (5,1))
dense1 =   LSTM(20, activation = "relu")(input1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(100, activation = "relu")(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)


# Compile and Fit and Early_Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 30, mode = "auto")
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train, y_train, epochs = 200, batch_size = 6, validation_data = (x_val, y_val), callbacks = early_stopping)


# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
y_test_predict = model.predict(x_test)
print("y_test_predict: ", y_test_predict.shape)
print("y_test_predict: \n", y_test_predict)
print("y_test: \n", y_test)

loss = model.evaluate(x_test, y_test, batch_size = 6)
rmse = RMSE(y_test, y_test_predict)
R2 = r2_score(y_test, y_test_predict)

print("loss: ", loss)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2)

result = model.predict(x_predict)
print("result: \n", result)

# result:
#  [[101.16037 ]
#  [102.13097 ]
#  [103.099815]
#  [104.06682 ]
#  [105.03202 ]]

# result: 
#  [[100.82014 ]
#  [101.794525]
#  [102.76765 ]
#  [103.73944 ]
#  [104.709946]]