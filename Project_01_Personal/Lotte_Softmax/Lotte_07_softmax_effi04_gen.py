import numpy as np
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, GaussianDropout
from tensorflow.keras.applications import MobileNet, EfficientNetB4
import tensorflow as tf
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#########데이터 로드
x_train = np.load("../data/lotte/lotte_data/train_x(192,192).npy")
y_train = np.load("../data/lotte/lotte_data/train_y(192,192).npy")

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

#########Augmentation
image_train_gen = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    rotation_range = 30
)

image_val_gen = ImageDataGenerator()

train_dataset = image_train_gen.flow(x_train,
                                     y_train,
                                     batch_size = 48,
                                     shuffle = True,
                                     seed = 77)

val_dataset = image_val_gen.flow(x_val,
                                 y_val,
                                 batch_size = 48,
                                 shuffle = True,
                                 seed = 777)


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size= 0.8, shuffle=True, random_state=66)

print(len(x_train))
print(y_train.shape)
print(len(x_val))
print(y_val.shape)


#########모델
eff = EfficientNetB4(input_shape = (192,192, 3), include_top = False, weights = 'imagenet')
eff.trainable = False

model = Sequential()
model.add(eff)

model.add(Conv2D(512, 3, 1, padding = 'same', activation = 'swish'))
model.add(BatchNormalization())
# model.add(GaussianDropout(0.1))

model.add(GlobalAveragePooling2D())

model.add(Flatten())
model.add(Dense(256, activation = 'swish'))
model.add(BatchNormalization())
# model.add(GaussianDropout(0.1))

model.add(Dense(128, activation = 'swish'))
model.add(BatchNormalization())
# model.add(GaussianDropout(0.1))

model.add(Dense(128, activation = 'swish'))
model.add(BatchNormalization())
# model.add(GaussianDropout(0.1))

model.add(Dense(1000, activation='softmax'))

model.summary()

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
optimizer = Adam(lr=0.01)
file_path = '../data/lotte/model/Lotte_model_07_effi04_aug.hdf5'
es = EarlyStopping(monitor='val_loss', patience=40, mode='min')
mc = ModelCheckpoint(file_path, monitor='val_accuracy', save_best_only=True, mode='auto', verbose=1)
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=15, verbose=1, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
# model.fit(x_train, y_train, batch_size = 32, epochs = 300, validation_data = (x_val, y_val), callbacks = [es,mc,rl])
model.fit_generator(
    train_dataset,
    steps_per_epoch = len(x_train)/48,
    epochs = 200,
    validation_data = val_dataset,
    validation_steps = len(x_val)/48,
    callbacks = [es, mc, rl]
)

###############Predict
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

file_path = '../data/lotte/model/Lotte_model_07_effi04_aug.hdf5'
model = load_model(file_path)
x_pred = np.load('../data/lotte/lotte_data/predict_x(192,192).npy')

result = model.predict(x_pred)

print(result.shape)

import pandas as pd
submission = pd.read_csv('../data/lotte/sample.csv')
submission['prediction'] = result.argmax(1)

submission.to_csv('../data/lotte/result/sample_07_effi04_aug(192,192).csv', index=False)
