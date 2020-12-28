from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. 데이터
x = np.array(range(1,101))
y = np.array(range(1,101))

# x_train = x[:60]                # 1 ~ 60
# x_val = x[60:80]                # 61 ~ 80
# x_test = x[80:]                 # 81 ~ 100
# 
# y_train = y[:60]                # 1 ~ 60
# y_val = y[60:80]                # 61 ~ 80
# y_test = y[80:]                 # 81 ~ 100

from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) # 무작위
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=False) -> 위와 다르게 순서대로 데이터 추출

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.2, shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.7, shuffle=False)
# 위 두가지 경우(size가 1보다 크거나 작은 경우) ERROR가 날까??? 정리
# size가 1을 넘는 경우 ERROR 발생, 하지만 size가 1보다 작은경우 가능
print("x_train: ", "\n", x_train)
print(x_train.shape)
print("\n")

print("x_test: ", "\n", x_test)
print(x_test.shape)
print("\n")

# train_test_split를 사용하여 val data 만들기
x_val, x_train, y_val, y_train = train_test_split(x_train, y_train, train_size=0.3, shuffle=False)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=False)
# 위 3가지는 같다.

print("x_train: ", "\n", x_train)
print(x_train.shape)
print("\n")

print("x_val: ", "\n", x_val)
print(x_val.shape)
print("\n")

# print(y_train.shape)
# print(y_test.shape)
# print(y_val.shape)


# 2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = "mse", optimizer= "adam", metrics = ["mae"])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

# 3. 평가 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("mae: ", mae)

y_predict = model.predict(x_test)
print("y_predict: ", "\n", y_predict)

# Shuffle = True , 보통 Shuffle = True가 False보다 성능이 좋다
# loss:  0.0009162042406387627
# mae:  0.02613486722111702

# Validation_split = 0.2 데이터양이 적다보니 val 데이터를 뺀후 훈련 횟수가 val 설정 전보다 적어져서 성능이 나빠질 수도 있다.
# loss:  0.00017840518557932228
# mae:  0.011711835861206055

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print("RMSE: ", RMSE(y_test, y_predict))
print("R2: ", r2)
'''