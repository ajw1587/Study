# 4번 카피
# CNN으로 딥하게 구성
# 2개의 모델을 만드는데 하나는 원칙적 오토인코더
# 다른 하나는 랜덤하게 만들고 싶은데로 히든을 구성
# 2개의 성능 비교

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

(x_train, _), (x_test, _) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1)/255.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Conv2DTranspose

# 1. 원칙대로
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(128, (2, 2), 1, padding = 'same', input_shape = (28, 28, 1)))
    model.add(Conv2D(32, (2, 2), 1, padding = 'same', activation = 'relu'))
    model.add(Dense(units = hidden_layer_size, activation = 'relu'))
    model.add(Conv2DTranspose(32, (2, 2), 1, padding = 'same', activation = 'relu'))
    model.add(Conv2DTranspose(128, (2, 2), 1, padding = 'same', activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    return model

# 2. 마음대로
# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Dense(units = hidden_layer_size, input_shape = (784, ),
#               activation = 'relu'))
#     model.add(Dense(units = 784, activation = 'sigmoid'))
#     return model

model = autoencoder(hidden_layer_size = 16)

model.summary()

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(x_train, x_train, epochs = 10)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize = (20, 7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap = 'gray')
    if i ==0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()