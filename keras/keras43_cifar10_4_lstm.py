import numpy as np
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train.shape: ", x_train.shape)       # (50000, 32, 32, 3)
print("y_train.shape: ", y_train.shape)       # (50000, 1)
print("x_test.shape: ", x_test.shape)         # (10000, 32, 32, 3)
print("y_test.shape: ", y_test.shape)         # (10000, 1)

print("x_train[0].shape", x_train[0].shape)   # (32, 32, 3)
print("y_train[0].shape", y_train[0].shape)   # (1,)

# x_train.reshape(50000, 1024, 3) <<<<이거는 실행이 되지 않는다.
x_train = x_train.reshape(50000, 96, 32)
x_test = x_test.reshape(10000, 96, 32)

x_train = x_train/255.
x_test = x_test/255.

# OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, LSTM

input1 = Input(shape = (96, 32))
dense1 = LSTM(50)(input1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
output1 = Dense(10, activation = 'softmax')(dense1)
model = Model(inputs = input1, outputs = output1)

model.summary()


# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 3, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.2, callbacks = es)


# Evaluate and Predict
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)