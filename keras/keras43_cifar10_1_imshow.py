import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train.shape: ", x_train.shape)       # (50000, 32, 32, 3)
print("y_train.shape: ", y_train.shape)       # (50000, 1)
print("x_test.shape: ", x_test.shape)         # (10000, 32, 32, 3)
print("y_test.shape: ", y_test.shape)         # (10000, 1)

print("x_train[0].shape", x_train[0].shape)   # (32, 32, 3)
print("y_train[0].shape", y_train[0].shape)   # (1,)
print(y_train[:50])

plt.imshow(x_train[0], 'hot')
plt.show()