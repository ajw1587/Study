# keras.io/api/layers/merging_layers/concatenate

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, concatenate

x = np.arange(20).reshape(2, 2, 5)
print(x)
# [[[ 0  1  2  3  4]
#   [ 5  6  7  8  9]]
#  [[10 11 12 13 14]
#   [15 16 17 18 19]]]
y = np.arange(20, 30).reshape(2, 1, 5)
print(y)
# [[[20 21 22 23 24]]
#  [[25 26 27 28 29]]]


z = Concatenate(axis=1)([x, y]) # Class
print(z)
# tf.Tensor(
# [[[ 0  1  2  3  4]
#   [ 5  6  7  8  9]
#   [20 21 22 23 24]]

#  [[10 11 12 13 14]
#   [15 16 17 18 19]
#   [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32)
print("\n")
print("\n")


b = concatenate([x, y], axis = 1) # Function
print(b)
# tf.Tensor(
# [[[ 0  1  2  3  4]
#   [ 5  6  7  8  9]
#   [20 21 22 23 24]]

#  [[10 11 12 13 14]
#   [15 16 17 18 19]
#   [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32)

# axis = 1 행
# axis = 0 열