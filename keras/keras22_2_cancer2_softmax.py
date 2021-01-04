# keras 21의 tensor1.py를 다중분류로 코딩하시오.

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer

# 1. 데이터 ===============================================
dataset = load_breast_cancer()
x = dataset.data                # (569, 30)
y = dataset.target              # Binary Data
# print(x.shape)
# print(y)

# y 값 Encoding 해주기
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = y.reshape(-1,1)
enc.fit(y)
y = enc.transform(y).toarray()
print(y)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)

# 2. 데이터 전처리==========================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# =========================================================
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 80)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 80)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 3. 모델 구성==============================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# =========================================================
model = Sequential()
model.add(Dense(10, input_shape = (30,)))
model.add(Dense(100, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(2, activation = "softmax"))

# 4. Compile and Fit ======================================
from tensorflow.keras.callbacks import EarlyStopping
# =========================================================
early_stopping = EarlyStopping(monitor = "loss", patience = 20, mode = "auto")
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["acc"])
model.fit(x_train, y_train, epochs = 100, batch_size = 6, validation_data = (x_val, y_val), callbacks = early_stopping)

# 5. Evaluate and Predict
loss, acc = model.evaluate(x_test, y_test, batch_size = 6)
y_predict = model.predict(x_test[-5:-1])

print("loss: ", loss)
print("acc: ", acc)
print("y_predict: \n", y_predict)
