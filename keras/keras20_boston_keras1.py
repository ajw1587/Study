# 2개의 파일을 만드시오.
# 1. EarlyStopping 을 적용하지 않은 최고의 모델
# 2. EarlyStopping 을 적용한 최고의 모델

from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
# 이걸로 만들어라!!


# EarlyStopping 적용
# 데이터
import numpy as np
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split = 0.2, seed = 77)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2, random_state = 77)

print("x_train.shape: ", x_train.shape)     # (323, 13)
print("y_train.shape: ", y_train.shape)     # (323,)
print("x_val.shape: ", x_val.shape)         # (81, 13)
print("y_val.shape: ", y_val.shape)         # (81,)
print("x_test.shape: ", x_test.shape)       # (102,13)
print("y_test.shape: ", y_test.shape)       # (102,)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_shape = (13,), activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(500, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(10,  activation = "relu"))
model.add(Dense(1))

# Compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 30, mode = "auto")
model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(x_train, y_train, batch_size = 3, epochs = 5000, validation_data = (x_val, y_val), callbacks = early_stopping)

# 평가 및 예측
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

loss, mae = model.evaluate(x_test, y_test, batch_size = 3)
y_predict = model.predict(x_test)
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("loss(mse): ", loss)
print("mae: ", mae)
print("RMSE: ", rmse)
print("R2_SORE: ", r2)

# loss(mse):  22.749813079833984
# mae:  2.8182480335235596
# RMSE:  4.769676315331989
# R2_SORE:  0.8215163246332474

# loss(mse):  14.63219928741455
# mae:  2.7454752922058105
# RMSE:  3.8252058559568907
# R2_SORE:  0.8852030606382431