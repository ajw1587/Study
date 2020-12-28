# 실습
# R2 음수가 아닌 0.5이하로 줄이기
# 1. 레이어는 인풋과 아웃풋을 포함 6개 이상
# 2. batch_size = 1
# 3. epochs = 100 이상
# 4. 데이터 조작 금지
# 5. 히든레이어의 노드 수는 10 이상

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import array

# 1. Data
x_train = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = array([11, 12, 13, 14, 15])
y_test = array([11, 12, 13, 14, 15])
x_pred = array([16, 17, 18])

# 2. model
model = Sequential()
model.add(Dense(10, input_dim = 1, activation = "relu"))
model.add(Dense(12000))
model.add(Dense(1200))
model.add(Dense(12000))
model.add(Dense(1200))
model.add(Dense(12000))
model.add(Dense(1200))
model.add(Dense(12000))
model.add(Dense(1200))
model.add(Dense(6000))
model.add(Dense(1200))
model.add(Dense(12000))
model.add(Dense(1))

# 3. compile and traning
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.2) # validation_split -> train 데이터에서 20%를 검증용 데이터로 사용

# 4. 예측 및 평가
results = model.evaluate(x_test, y_test, batch_size = 1)
print("results(mes, mae): ", results)

y_predict = model.predict(x_test)
# print("y_predict: ", "\n", y_predict)

# 사이킷런 (sklearn)
from sklearn.metrics import mean_squared_error

def RMSE(y_test , y_predict): # y 예측값과 실제 y갑의 RMSE값
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("mean_squared_error: ", mean_squared_error(y_predict, y_test))
print("RMSE: ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score # R2(결정계수)
r2 = r2_score(y_test, y_predict)
print("R2: ", r2)