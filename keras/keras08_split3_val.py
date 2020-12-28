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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8) # 무작위
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=False) -> 위와 같이 순서대로 데이터 추출

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

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
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

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