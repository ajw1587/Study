import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data_path = '../data/sign_image/sign_language01/Sign_Language_for_Alphabets'
data_save_path = '../data/sign_image/sign_language01/numpy_data'
batch = 64

# ImageGenerator 설정
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 0.4
)

except_datagen = ImageDataGenerator(
    rescale = 1./255
)

# ImageGenerator 적용
data_flow = train_datagen.flow_from_directory(
    data_path, target_size = (64, 64),
    class_mode = 'categorical',
    batch_size = 39000,
    seed = 77
)

# x_data = tf.image.rgb_to_grayscale(data_flow[0][0])

np.save(data_save_path + '/sign_language01_color_x.npy', data_flow[0][0])
# np.save(data_save_path + '/sign_language01_y.npy', data_flow[0][1])

print(data_flow[0][0].shape)
print(data_flow[0][1].shape)
print(type(data_flow))
print(data_flow.class_indices)
# (39000, 64, 64, 1)
# (39000, 26)
# <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 
# 22, 'x': 23, 'y': 24, 'z': 25}

'''
# Load Data
x_train = np.load(data_save_path + '/sign_language01_x.npy')
y_train = np.load(data_save_path + '/sign_language01_y.npy')

print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
# x_train.shape:  (39000, 64, 64, 1)
# y_train.shape:  (39000, 26)

# Data Preprocessing
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 77)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 77)

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)
# (24960, 64, 64, 1)
# (7800, 64, 64, 1)
# (6240, 64, 64, 1)
# (24960, 26)
# (7800, 26)
# (6240, 26)

# Model
opti = Adam(learning_rate = 0.01)
es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'auto')
cp = ModelCheckpoint('../data/modelcheckpoint/sign_language/sign_language_model_new3.hdf5')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 50, mode = 'auto')
def my_model():
    input1 = Input(shape = (64, 64, 1))
    x = Conv2D(128, (2, 2), 1, padding = 'same', activation = 'relu')(input1)
    x = MaxPooling2D(pool_size = (3, 3))(x)
    # x = Dropout(drop)(x)
    x = Conv2D(64, (2, 2), 1, padding = 'same', activation = 'relu')(x)
    x = Conv2D(64, (2, 2), 1, padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (3, 3))(x)
    # x = Dropout(drop)(x)
    x = Conv2D(32, (2, 2), 1, padding = 'same', activation = 'relu')(x)
    x = Conv2D(32, (2, 2), 1, padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (3, 3))(x)
    # x = Dropout(drop)(x)

    # x = Conv2D(128, (3, 3), 1, padding = 'same', activation = 'relu')(x)
    # x = Dropout(drop)(x)
    # x = Conv2D(64, (3, 3), 1, padding = 'same', activation = 'relu')(x)
    # x = Dropout(drop)(x)
    # x = Conv2D(32, (3, 3), 1, padding = 'same', activation = 'relu')(x)
    # x = Dropout(drop)(x)

    # x = Conv2D(128, (2, 2), 1, padding = 'same', activation = 'relu')(x)
    # x = Dropout(drop)(x)
    x = Conv2D(64, (2, 2), 1, padding = 'same', activation = 'relu')(x)
    # x = Dropout(drop)(x)
    x = Conv2D(32, (2, 2), 1, padding = 'same', activation = 'relu')(x)
    # x = Dropout(drop)(x)

    x = Flatten()(x)
    # x = Dense(128, activation = 'relu')(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation = 'relu')(x)
    # x = Dropout(drop)(x)
    x = Dense(16, activation = 'relu')(x)
    output1 = Dense(26, activation = 'softmax')(x)
    model = Model(inputs = input1, outputs = output1)

    model.compile(loss = 'categorical_crossentropy', optimizer = opti, metrics = ['acc'])

    model.summary()
    return model

model = my_model()

hist = model.fit(x_train, y_train,
                 epochs = 1000,
                 batch_size = 32,
                 validation_data = (x_val, y_val),
                 callbacks = [es, cp, reduce_lr])
acc = hist.history['acc']
loss = hist.history['loss']
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

print('acc: ', acc[-1])
print('loss: ', loss[-1])
print('val_acc: ', val_acc[-1])
print('val_loss: ', val_loss[-1])

model2 = load_model('../data/modelcheckpoint/sign_language/sign_language_model_new3.hdf5')

def RMSE(y_test, y_test_predict):
    return np.sqrt(mean_squared_error(y_test, y_test_predict))

y_pred = model2.predict(x_test)
rmse = RMSE(y_test, y_pred)

print('RMSE: ', rmse)
'''
# sign_language_model_new3.hdf5
# acc: 0.037860576063394547
# loss: 3.2587742805480957
# val_acc: 0.03701923042535782
# val_loss: 3.2590649127960205
# RMSE: 0.19231465