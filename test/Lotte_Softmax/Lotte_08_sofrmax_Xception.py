import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNet, ResNet50, ResNet101, Xception
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime 

submission = pd.read_csv('../data/lotte/sample.csv')
# print(submission.shape) # (72000, 2)

start_now = datetime.datetime.now()

### npy load
x_data = np.load("../data/lotte/lotte_data/train_x(192,192).npy", allow_pickle=True)
print(x_data.shape)    # (48000, 100, 100, 3)
y_data = np.load("../data/lotte/lotte_data/train_y(192,192).npy", allow_pickle=True)
print(y_data.shape)    # (48000,)
x_pred = np.load('../data/lotte/lotte_data/predict_x(192,192).npy', allow_pickle=True)
print(x_pred.shape)     # (72000, 100, 100, 3)


#1. DATA
# preprocess
x_data = preprocess_input(x_data)
x_pred = preprocess_input(x_pred)

y_data = to_categorical(y_data)

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    width_shift_range= 0.05,
    height_shift_range= 0.05
)

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, train_size=0.9, shuffle=True, random_state=42)
print(x_train.shape, x_valid.shape)  # (43200, 100, 100, 3) (4800, 100, 100, 3)
print(y_train.shape, y_valid.shape)  # (43200, 1000) (4800, 1000)

dropout_rate = 0.2

def my_model () :
    transfer = Xception(weights="imagenet", include_top=False, input_shape=(192,192, 3))
    for layer in transfer.layers:
            layer.trainable = True
    top_model = transfer.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Flatten()(top_model)
    top_model = Dense(2048, activation="swish")(top_model)
    top_model = Dropout(dropout_rate) (top_model)
    top_model = Dense(1000, activation="softmax")(top_model)
    model = Model(inputs=transfer.input, outputs = top_model)
    return model

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.06)
path = '../data/lotte/model/Lotte_model_08_Xception.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

batch = 16
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
valid_generator = test_datagen.flow(x_valid, y_valid, batch_size=batch)
pred_generator = x_pred

model = my_model()
# model.summary()

#3. Compile, Train, Evaluate
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit_generator(train_generator, epochs=200, steps_per_epoch = len(x_train) // batch ,
    validation_data=valid_generator, validation_steps=10 , callbacks=[es, lr, cp])
model.save_weights('../data/lotte/weights/Lotte_08_Xception_weights.h5')
result = model.evaluate(valid_generator, batch_size=batch)
print("loss ", result[0])   # 0.00693
print("acc ", result[1])    # 0.9979166


#4. Predict
# model = load_model('../data/LPD_competition/cp/cp_0324_2_resnet.hdf5')
model.load_weights('../data/lotte/weights/Lotte_08_Xception_weights.h5')

print(">>>>>>>>>>>>>>>> predict >>>>>>>>>>>>>> ")
# tta 나만 터져?

result = model.predict(pred_generator, verbose=True)

# save
print(result.shape) # (72000, 1000)
print(np.argmax(result, axis = 1))
result_arg = np.argmax(result, axis = 1)

submission['prediction'] = result_arg
submission.to_csv('../data/lotte/result/sample_08_Xception.csv', index=False)
# /sub_0324_2 > score 70.389    ***
# /sub_0324_3 > score 70.471    ***

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >>  3:06:57

