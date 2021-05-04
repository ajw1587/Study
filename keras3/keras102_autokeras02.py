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

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=6)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3, factr = 0.5, verbose = 2)
ck = ModelCheckpoint('./autokeras_save_model/', save_weights_only = True,                              # 같은 경로에서만 저장이 되네???
                     save_best_only = True, monitor = 'val_loss', verbose = 1)

model = ak.ImageClassifier(
    overwrite = True,
    max_trials = 2,
    loss = 'mse',
    metrics = ['acc']
)
# model.summary()     # ERROR 발생

model.fit(x_train, y_train, epochs = 1,
          validation_split = 0.2, # validation_split: default: 0.2
          callbacks = [es, lr, ck])

results = model.evaluate(x_test, y_test)

print(results)

model.summary()