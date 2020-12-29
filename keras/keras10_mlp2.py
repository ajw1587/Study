import numpy as np
from sklearn.model_selection import train_test_split

x = np.array([range(100), range(301,401), range(1, 101)])
y = np.array(range(711, 811))

# 열은 특징, 행은 속성값 즉, 행무시 열우선
# print(x)
# print(x.shape)      # (3,100)
# print(y)
# print(y.shape)      # (100,)

# (2,10) -> (10,2)
# print(x.T)                  # 3차원: 0, 2차원: 1, 1차원: 2
# print(np.swapaxes(x,0,1))   # 3차원: 0, 2차원: 1, 1차원: 2
x = np.transpose(x)
# print(x)                      # 3차원: 0, 2차원: 1, 1차원: 2

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2)

print("x_train: ", x_train)
print("x_val: ", x_val)
print("x_test: ", x_test)

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 3))     # input_dim = 열(컬럼) 갯수
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 컴파일
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train ,y_train, epochs = 100, batch_size = 1, validation_data = (x_val, y_val))

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