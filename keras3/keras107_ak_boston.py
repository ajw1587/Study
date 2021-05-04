import numpy as np
import autokeras as ak
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

x_train, x_test, y_train, y_test = train_test_split(load_boston().data, load_boston().target, train_size = 0.8, random_state = 77)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model = ak.StructuredDataRegressor(overwrite = True,
                                   max_trials = 1,
                                   loss = 'mse',
                                   metrics = ['mae'])

es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience=6)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience=3, factr = 0.5, verbose = 2)
model.fit(x_train, y_train, epochs = 1, callbacks = [es, lr], validation_split = 0.2)

# SAVE Best Model
# model = model.export_model()
best_model = model.tuner.get_best_model()
best_model.save('F:/autokeras/save_model/best_boston.tf')                   # 왜 hdf5, h5 파일 형식으로는 저장이 되지 않지???

# LOAD Best Model
best_model = load_model('F:/autokeras/save_model/best_boston.tf')
results = best_model.evaluate(x_test, y_test)
print('results: ', results)
best_model.summary()