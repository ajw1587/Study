import os
import os, glob, numpy as np
from PIL import Image
import numpy as np
from numpy import asarray
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
import tensorflow as tf
import scipy.signal as signal
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

'''
#########데이터 로드
x = np.load("../data/lotte/train_x(256,256).npy")
y = np.load("../data/lotte/train_y(256,256).npy")

print(x.shape)
print(y.shape)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size= 0.8, shuffle=True, random_state=66)

inputs = Input(shape = (256, 256, 3))
x = Conv2D(256, 4, padding="SAME", activation='relu')(inputs)
x = MaxPooling2D(2)(x)
x = BatchNormalization()(x)

x = Conv2D(128, 2, padding="SAME", activation='relu')(x)
x = MaxPooling2D(2)(x)
x = BatchNormalization()(x)

x = Conv2D(64, 2, padding="SAME", activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(32, 2, padding="SAME", activation='relu')(x)
x = BatchNormalization()(x)

x = Flatten()(x)

x = Dense(1200, activation='relu')(x)
x = BatchNormalization()(x)

x = Dense(1100, activation='relu')(x)
x = BatchNormalization()(x)

outputs = Dense(1000, activation='softmax')(x)
model = Model(inputs = inputs, outputs = outputs)

# model = Sequential()
# model.add(effi)
# model.add(Flatten())
# model.add(Dense(1000, activation='softmax'))

# model.summary()

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, mode='min')
file_path = '../data/lotte/Lotte_model_04.hdf5'
mc = ModelCheckpoint(file_path, monitor='val_accuracy',save_best_only=True,mode='max',verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss',factor=0.3,patience=15,verbose=1,mode='min')
model.fit(x_train, y_train, batch_size=16, epochs=200, validation_data=(x_val, y_val),callbacks=[es,mc,rl])
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
loss, acc = model.evaluate(x_val, y_val)
'''

model = load_model('../data/lotte/Lotte_model_04.hdf5')
x_pred = np.load('../data/lotte/predict_x(256,256).npy')

result = model.predict(x_pred)

print(result.shape)

import pandas as pd
submission = pd.read_csv('../data/lotte/sample.csv')
submission['prediction'] = result.argmax(1)

submission.to_csv('../data/lotte/sample_04.csv', index=False)

# 63.292