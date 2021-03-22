import numpy as np
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.applications import EfficientNetB7
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''
#########데이터 로드
x_train = np.load("../data/lotte/lotte_data/train_x(128,128).npy")
y_train = np.load("../data/lotte/lotte_data/train_y(128,128).npy")

# print(x.shape)
# print(y.shape)

#########Augmentation
image_train_gen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1
)

image_val_gen = ImageDataGenerator(
    rescale = 1./255
)

train_dataset = image_train_gen.flow(x_train,
                                     y_train,
                                     batch_size = 48,
                                     shuffle = True,
                                     seed = 77)

val_dataset = image_val_gen.flow(x_train,
                                 y_train,
                                 batch_size = 48,
                                 shuffle = True,
                                 seed = 777)


# from sklearn.model_selection import train_test_split
# x_train, x_val, y_train, y_val = train_test_split(train_dataset[0][0], train-dataset[0][1], train_size= 0.8, shuffle=True, random_state=66)


#########모델
effi = EfficientNetB7(include_top = False, weights = 'imagenet', input_shape = (128, 128, 3))
effi.trainable = False

model = Sequential()
model.add(effi)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(1000, activation='softmax'))

model.summary()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
optimizer = Adam(lr=0.001)
file_path = '../data/lotte/model/Lotte_model_07_Effi07.hdf5'
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
mc = ModelCheckpoint(file_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, verbose=1, mode='min')

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
# model.fit(x_train, y_train, batch_size = 16, epochs = 100, validation_data = (x_val, y_val), callbacks = [es,mc,rl])
model.fit_generator(
    train_dataset,
    steps_per_epoch = 1000,
    epochs = 100,
    validation_data = val_dataset,
    validation_steps = 20,
    callbacks = [es, mc, rl]
)
'''
###############Predict
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

file_path = '../data/lotte/model/Lotte_model_07_Effi07.hdf5'
model = load_model(file_path)
x_pred = np.load('../data/lotte/lotte_data/predict_x(128,128).npy')

result = model.predict(x_pred)

print(result.shape)

import pandas as pd
submission = pd.read_csv('../data/lotte/sample.csv')
submission['prediction'] = result.argmax(1)

submission.to_csv('../data/lotte/result/sample_07_Effi07.csv', index=False)
