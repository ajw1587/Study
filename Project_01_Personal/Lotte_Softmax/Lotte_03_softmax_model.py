import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, MaxPooling2D, Conv2D, Flatten, Dropout
from tensorflow.keras.applications import EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetB4
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

################## Data
x_train = np.load('../data/lotte/train_x(256,256).npy')
y_train = np.load('../data/lotte/train_y(256,256).npy')
x_test = np.load('../data/lotte/predict_x(256,256).npy')

# x_aug_train = np.load('../data/lotte/train_aug_x(128,128).npy')
# y_aug_train = np.load('../data/lotte/train_aug_y(128,128).npy')

# print(x_train.shape)        # (48000, 64, 64, 3)
# print(x_aug_train.shape)    # (48000, 64, 64, 3)
# print(y_train.shape)        # (48000, 1000)
# print(y_aug_train.shape)    # (48000, 1000)
# print(x_test.shape)         # (72000, 64, 64, 3)

# x_train = np.concatenate((x_train, x_aug_train[:10000, :]), axis = 0)
# y_train = np.concatenate((y_train, y_aug_train[:10000, :]), axis = 0)
# x_train = x_aug_train
# y_train = y_aug_train

# print('x_train_type: ', type(x_train))      # <class 'numpy.ndarray'>
# print('x_train_shape: ', x_train.shape)     # (96000, 64, 64, 3)
# print('y_train_type: ', type(y_train))      # <class 'numpy.ndarray'>
# print('y_train_shape: ', y_train.shape)     # (96000, 1000)
# print('x_test_type: ', type(x_test))        # <class 'numpy.ndarray'>
# print('x_test_shape: ', x_test.shape)       # (72000, 64, 64, 3)


################## Split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

# print('x_train.shape: ', x_train.shape)     # (61440, 64, 64, 3)
# print('x_test.shape: ' , x_test.shape)      # (19200, 64, 64, 3)
# print('x_val.shape: '  , x_val.shape)       # (15360, 64, 64, 3)
# print('y_train.shape: ', x_train.shape)     # (61440, 64, 64, 3)
# print('y_test.shape: ' , x_test.shape)      # (19200, 64, 64, 3)
# print('y_val.shape: '  , x_val.shape)       # (15360, 64, 64, 3)


################## Model
efficientnetb4 = EfficientNetB4(include_top = False, weights = 'imagenet', input_shape = (256, 256, 3))
efficientnetb4.trainable = False

model = Sequential()
model.add(Conv2D(1200, 2, (3, 3), padding = 'same', activation = 'relu', input_shape = (256, 256, 3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(1500, 2, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(1200, 2, (3, 3), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Conv2D(1000, 2, (3, 3), padding = 'same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(1200, activation = 'relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1100, activation = 'relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1000, activation = 'softmax'))

model.summary()

################## Model compile and fit
model_save_path = '../data/lotte/Lotte_Model_Mymodel.hdf5'
es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 10)
cp = ModelCheckpoint(model_save_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(optimizer = Adam(learning_rate = 0.1),
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

hist = model.fit(x_train, y_train, 
                 batch_size = 20, 
                 epochs = 100, 
                 validation_data = (x_val, y_val), 
                 callbacks = [es, cp, reduce_lr])

loss, acc = model2.evaluate(x_test, y_test)
y_test_predict = model.predict(x_test)
rmse = RMSE(y_test, y_test_predict)
print("loss: ", loss)
print("acc: ", acc)
print("RMSE: ", rmse)
# 똥값이 나온다!!!!
# 실패 모델!


# predict
model2 = load_model(model_save_path)