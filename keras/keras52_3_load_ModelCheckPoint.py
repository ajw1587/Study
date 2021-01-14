# model save, weight save 비교하기

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.

# OneHotEncoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# model = Sequential()
# model.add(Conv2D(filters = 100, kernel_size = (2,2), strides = 1, padding = 'same', input_shape = (28, 28, 1)))
# model.add(MaxPooling2D(pool_size = 2))
# model.add(Conv2D(100, 2, padding = 'same'))         # padding = 'valid': 패딩 사용 안함
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(10, activation = 'softmax'))

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
# k52_1_mnist_??? => k52_1_MCK_val_loss.hdf5 이름을 바꿔줄것
# es = EarlyStopping(monitor = 'loss', patience =3, mode = 'auto')
# cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# 4-1. load_weights 평가 예측 -> 모델구성 필요O
# model.load_weights('../data/h5/k52_1_weight.h5')
# result = model.evaluate(x_test, y_test, batch_size = 32)
# print("가중치_loss: ", result[0])
# print("가중치_accuracy: ", result[1])
# 가중치_loss:  0.1292164921760559
# 가중치_accuracy:  0.9589999914169312

# 4-2. load_model 평가 예측 -> 모델구성 필요X
# model2 = load_model('../data/h5/k52_1_model2.h5')
# result2 = model.evaluate(x_test, y_test, batch_size = 32)
# print("로드모델_loss: ", result2[0])
# print("로드모델_accuracy: ", result2[1])
# 로드모델_loss:  0.1292164921760559
# 로드모델_accuracy:  0.9589999914169312

# 4-3. load_modelcheckpoint 평가 예측 -> 모델구성 필요X, modelcheckpoint를 이용하여 최적의 가중치 보유
model = load_model('../data/modelcheckpoint/k52_1_mnist_checkpoint.hdf5')
result = model.evaluate(x_test, y_test, batch_size = 32)
print("로드체크포인트_loss: ", result[0])
print("로드체크포인트_accuracy: ", result[1])
# 로드체크포인트_loss:  0.11277369409799576
# 로드체크포인트_accuracy:  0.9639999866485596