# 인공지능계의 hello world라 불리는 mnist!!!
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape: ", x_train.shape)       # (60000, 28, 28)
print("y_train.shape: ", y_train.shape)       # (60000,)
print("x_test.shape: ", x_test.shape)         # (10000, 28, 28)
print("y_test.shape: ", y_test.shape)         # (10000,)

print("x_train[0].shape", x_train[0].shape)   # (28, 28)
print("y_train: \n ", y_train[:10])
# plt.imshow(x_train[0], 'hot')
# plt.show()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.
# (x_test.reshape(x_test.shape[0], x_test.shaep[1], x_test.shaep[2], 1))

# OneHotEncoding
# 여러분이 하시오!
# from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 100, kernel_size = (2,2), strides = 1, padding = 'same', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(100, 2, padding = 'same'))         # padding = valid: 패딩 사용 안함
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 실습!! 완성하시오!!!
# 지표는 acc
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience =3, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, batch_size = 10, epochs = 100, validation_split = 0.2, callbacks = es)

# 응용
# y_test 10개와 y_test 10개를 출력하시오.
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)
# print("y_test[:10]: \n", y_test[:10])
# print("y_predict[:10]: \n", y_predict[:10])

# y_test[:10] = (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
# y_pred[:10] = (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

# loss:  0.11771859228610992
# acc:  0.9699000120162964

# loss:  0.11285488307476044
# acc:  0.9689000248908997