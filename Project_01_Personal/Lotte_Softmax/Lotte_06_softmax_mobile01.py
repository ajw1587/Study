import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.applications import EfficientNetB7, MobileNetV2
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
'''
#########데이터 로드
x = np.load("../data/lotte/lotte_data/train_x(256,256).npy")
y = np.load("../data/lotte/lotte_data/train_y(256,256).npy")
x_aug = np.load('../data/lotte/lotte_data/train_x(128,128)_aug.npy')
y_aug = np.load('../data/lotte/lotte_data/train_y(128,128)_aug.npy')

# x = np.concatenate((x, x_aug[:10000, :]), axis = 0)
# y = np.concatenate((y, y_aug[:10000, :]), axis = 0)

print(x.shape)      # (58000, 128, 128, 3)
print(y.shape)      # (58000, 1000)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size= 0.8, shuffle=True, random_state=66)


#########모델
mobile = MobileNetV2(input_shape = (256, 256, 3), include_top = False, weights = 'imagenet')
mobile.trainable = True

model = Sequential()
model.add(mobile)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(4048, activation= 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='softmax'))

model.summary()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
optimizer = Adam(lr=0.001)
file_path = '../data/lotte/model/Lotte_model_06_Mobile.hdf5'
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
mc = ModelCheckpoint(file_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1, mode='min')

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 16, epochs = 100, validation_data = (x_val, y_val), callbacks = [es,mc,rl])
'''

###############Predict
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

file_path = '../data/lotte/model/Lotte_model_06_Mobile.hdf5'
model = load_model(file_path)
x_pred = np.load('../data/lotte/lotte_data/predict_x(256,256).npy')

result = model.predict(x_pred)

print(result.shape)

import pandas as pd
submission = pd.read_csv('../data/lotte/sample.csv')
submission['prediction'] = result.argmax(1)

submission.to_csv('../data/lotte/result/sample_06_Mobile.csv', index=False)
