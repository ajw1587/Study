# 주말과제
# dense 모델로 구성 input_shape = (28*28, )

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape: \n", x_train.shape)               # (60000, 28, 28)
print("y_train.shape: \n", y_train.shape)               # (60000, )
print("x_test.shape: \n", x_test.shape)                 # (10000, 28, 28)
print("y_test.shape: \n", y_test.shape)                 # (10000, )

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 70)
print(x_train.shape)
print(x_test.shape)

x_train = x_train/255.
x_test = x_test/255.
x_val = x_val/255.

# OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# 모델 작성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

model = Sequential()
model.add(Dense(100, input_shape = (28, 28), activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 2, batch_size = 100, validation_data = (x_val, y_val), callbacks = es)

# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error

loss, acc = model.evaluate(x_test, y_test, batch_size = 100)
y_test_predict = model.predict(x_test)

print(y_test_predict[0])
print('loss: ', loss)
print('acc: ', acc)