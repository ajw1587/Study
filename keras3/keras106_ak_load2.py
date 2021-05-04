# https://autokeras.com/tutorial/image_classification/
import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# One Hot 해도 되고 안해도 된다.
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model

model1 = load_model('F:/autokeras/save_model/aaa.hdf5')
model1.summary()
print('\n')
model2 = load_model('F:/autokeras/save_model/best_aaa.hdf5')
model2.summary()

results1 = model1.evaluate(x_test, y_test)
print('results1: ', results1)

best_results = model2.evaluate(x_test, y_test)
print('best_results: ', best_results)