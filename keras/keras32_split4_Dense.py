# 과제 및 실습  Dense
# EarlyStopping, 전처리 등등 지금까지 배운 내용 다 넣기
# 데이터는 1~ 100 / size: 5
#     x               y
# 1,2,3,4,5           6
# ...
# 95,96,97,98,99    100

# predict 만들것
# 96,97,98,99,100 -> 101
# ...
# 100,101,102,103,104 -> 105
# 예상 predict는 (101, 102, 103, 104, 105)
# keras32_split3_LSTM.py와 비교

# 데이터
import numpy as np
def data_split(data, size):
    arr = []
    for i in range(len(data) - size + 1):
        subset = data[i : i+size]
        arr.append(subset)
        # arr.append(i for i in subset)
    return np.array(arr)

data = np.array(range(1, 101))      # 1~ 100
dataset = data_split(data, 6)
print("dataset.shape: ", dataset.shape)     # (95, 6)

x = dataset[:, :5]      # (95,5)
y = dataset[:, 5:]      # (95,1)
print("x.shape: ", x.shape)
print("y.shape: ", y.shape)


# 데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 70)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 70)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)


# 예측 데이터
data2 = np.array(range(96, 106))
dataset2 = data_split(data2, 6)     # (5, 6)
print("dataset2.shape: ", dataset2.shape)

x_predict = dataset2[:, :5]
y_predict = dataset2[:, 5:]

x_predict = scaler.transform(x_predict)


# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

input1 = Input(shape = (5,))
dense1 =  Dense(20, activation = "relu")(input1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(100, activation = "relu")(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)


# Compile and Fit and EarluStopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 30, mode = "auto")
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train, y_train, epochs = 200, batch_size = 6, validation_data = (x_val, y_val), callbacks = early_stopping)


# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

y_test_predict = model.predict(x_test)
loss = model.evaluate(x_test, y_test, batch_size = 6)
rmse = RMSE(y_test, y_test_predict)
R2 = r2_score(y_test, y_test_predict)
print("loss: ", loss)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2)

result = model.predict(x_predict)
print("result: \n", result)

# result:
# [[101.000015]
#  [102.      ]
#  [103.00001 ]
#  [103.999985]
#  [105.      ]]

# result: 
#  [[101.00001 ]
#  [101.99999 ]
#  [103.000015]
#  [103.99999 ]
#  [105.00001 ]]