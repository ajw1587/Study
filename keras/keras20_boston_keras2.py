# 2개의 파일을 만드시오.
# 1. EarlyStopping 을 적용하지 않은 최고의 모델
# 2. EarlyStopping 을 적용한 최고의 모델

from tensorflow.keras.datasets import boston_housing
import numpy as np
# 이걸로 만들어라!!


# EarlyStopping 미적용
# 데이터
from sklearn.model_selection import train_test_split
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(train_split = 0.8, test_split = 0.2, seed = 77)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2, random_state = 77)

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

# compile and fit
model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(x_train, y_train, batch_size = 3, epochs = 200, validation_data = (x_val, y_val))

# 평가 및 예측
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
y_predict = model.predict(x_test)
rmse = RMSE(y_test, y_predict)
R2_SCORE = r2_score(y_test, y_predict)
loss, mae = model.evaluate(x_test, y_test, batch_size = 3)

print("loss(mse): ", loss)
print("mae: ", mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2_SCORE)