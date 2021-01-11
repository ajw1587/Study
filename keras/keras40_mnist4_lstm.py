# 주말과제
# LSTM 모델로 구성 input_shape = (28*28, )

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

# 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 70)

print(x_train.shape)          # (60000, 28, 28)
print(y_train.shape)          # (60000,)
print(x_test.shape)           # (10000, 28, 28)
print(y_test.shape)           # (10000,)

x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.

# OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Flatten

input1 = Input(shape = (28, 28))
dense1 = LSTM(100)(input1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
output1 = Dense(10, activation = 'softmax')(dense1)
model = Model(inputs = input1, outputs = output1)

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 3, batch_size = 32, validation_data = (x_val, y_val), callbacks = es)

# Evaluate and Predict
loss, acc = model.evaluate(x_val, y_val, batch_size = 32)
y_test_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)
print("y_test_predict[0]: ", y_test_predict[0])