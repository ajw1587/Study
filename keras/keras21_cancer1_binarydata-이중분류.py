import numpy as np
from sklearn.datasets import load_breast_cancer
# 1. 데이터
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape)      # (569, 30)
# print(y.shape)      # (569,)
# print(x[:5])
# print(y)

# 2. 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 70)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 70)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 3. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation = "relu", input_shape = (30,)))
model.add(Dense(100, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(400, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

# 4. Compile and Train
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['acc'])
model.fit(x, y, epochs = 150, validation_data = (x_val, y_val), batch_size = 3)
loss = model.evaluate(x, y)
print(loss)

# 실습1. loss 0.985 이상
# 실습2. y_predict 출력
pre = x[-5:-1]
y_predict = model.predict(pre)
print(x_val[-5:-1])
print(y_predict)

# [0.1164085790514946, 0.9507908821105957]
# [[7.3419528e-06]
#  [6.5521534e-05]
#  [5.5218562e-02]
#  [3.3050358e-06]]