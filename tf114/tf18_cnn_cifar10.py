import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

unique, counts = np.unique(y_train, return_counts = True)
print(np.array((unique, counts)))
# [[   0    1    2    3    4    5    6    7    8    9]  -> 10개
#  [5000 5000 5000 5000 5000 5000 5000 5000 5000 5000]]

x_train = x_train/255.
x_test = x_test/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)        # (50000, 32, 32, 3)
print(y_train.shape)        # (50000, 10)
print(x_test.shape)         # (10000, 32, 32, 3)
print(y_test.shape)         # (10000, 10)

# x, y, w, b
x = tf.compat.v1.placeholder('float32', [None, 32*32*3])
y = tf.compat.v1.placeholder('float32', [None, 32*32*3])

w = tf.compat.v1.get_variable('w1', shape = )