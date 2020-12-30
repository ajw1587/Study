import numpy as np
import tensorflow as tf

#1. data
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. model making
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim = 1, activation= "linear"))
model.add(Dense(3, activation= "linear"))
model.add(Dense(4))
model.add(Dense(1))

model.summary()
# Model: "sequential"   
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 5)                 10         이전노드+바이어스(1개) = 2,   2 * 현재 layer 노드수(5) = 10, 바이어스만큼 한번더 연산
# _________________________________________________________________
# dense_1 (Dense)              (None, 3)                 18         이전노드+바이어스(1개) = 6,   6 * 현재 layer 노드수(3) = 18, 바이어스만큼 한번더 연산
# _________________________________________________________________
# dense_2 (Dense)              (None, 4)                 16         이전노드+바이어스(1개) = 4,   4 * 현재 layer 노드수(4) = 6, 바이어스만큼 한번더 연산
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 5          이전노드+바이어스(1개) = 5,   5 * 현재 layer 노드수(1) = 18, 바이어스만큼 한번더 연산
# =================================================================
#                              none = 행무시              바이어스 값

# Total params: 49
# Trainable params: 49
# Non-trainable params: 0
# _________________________________________________________________