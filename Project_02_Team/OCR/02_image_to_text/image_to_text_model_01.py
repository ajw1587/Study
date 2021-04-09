import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
import cv2 as cv
import glob
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB4

# Check Data Length
img_path = glob.glob('F:/Team Project/OCR/Image_to_Text_model/image-data/hangul-images/*.jpeg')
print(len(img_path))
# 37600ea Image Data

# Read Data
first_path = 'F:/Team Project/OCR/Image_to_Text_model/image-data/hangul-images/hangul_'
last_path = '.jpeg'
img_list = []
for i in range(1, len(img_path) + 1):
    img = cv.imread(first_path + str(i) + last_path, cv.IMREAD_GRAYSCALE)
    img_list.append(img)
x_train = np.array(img_list)


# Read Data
x_train = pd.read_csv()

# Model
eff04 = EfficientNetB4(include_top = False, weights = 'imagenet', input_shape = (64, 64, 1))
initial_model = eff04
last = eff04.output

x = Flatten()(last)
x = Dense(32)(x)
x = Dense(16)(x)
output1 = Dense(10, activation = 'softmax')(x)
model = Model(inputs = initial_model.input, outputs = output1)


# dense = Conv2D(256, 2, 1, padding = 'same', activation = 'relu')(input)
# dense = MaxPooling2D(pool_size = (2, 2))(dense)
# dense = BatchNormalization()(dense)
# dense = Dropout(0.3)(dense)

# dense = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(input)
# dense = MaxPooling2D(pool_size = (2, 2))(dense)
# dense = BatchNormalization()(dense)
# dense = Dropout(0.3)(dense)

# dense = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(input)
# dense = MaxPooling2D(pool_size = (2, 2))(dense)
# dense = BatchNormalization()(dense)
# dense = Dropout(0.3)(dense)

# flatten = Flatten()(dense)

hist = model.compile(optimizer = Adam(learning_rate = 0.001),
                     loss = 'categorical_crossentropy',
                     metrics = ['acc'])
model.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_val, y_val))


result = model.evaluate(x_test, y_test, batch_size = 256)
y_predict = model.predict(x_test)