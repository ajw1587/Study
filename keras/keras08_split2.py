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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6) # 무작위
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
model.fit(x_train, y_train, epochs=100)

# 3. 평가 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("mae: ", mae)

y_predict = model.predict(x_test)
print("y_predict: ", "\n", y_predict)

# Shuffle = True
# loss:  0.0009162042406387627
# mae:  0.02613486722111702