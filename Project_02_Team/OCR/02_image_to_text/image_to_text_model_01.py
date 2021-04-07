import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Flatten

# Data
img = cv.imread('F:/Team Project/OCR/Text_detection/image_data/test_picture/test2/test0_0.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, dsize = (128, 128))
# cv.imshow('img', img)
# cv.moveWindow('img', 300, 300)
# cv.waitKey(0)
# cv.destroyAllWindows()

img = img.reshape(img.shape[0], img.shape[1], 1)
print(img.shape)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()


input = Input(shape = (128, 128, 1))
dense = Conv2D(256, 2, 1, padding = 'same', activation = 'relu')(input)
dense = MaxPooling2D(pool_size = (2, 2))(dense)
dense = BatchNormalization()(dense)
dense = Dropout(0.3)(dense)

dense = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(input)
dense = MaxPooling2D(pool_size = (2, 2))(dense)
dense = BatchNormalization()(dense)
dense = Dropout(0.3)(dense)

dense = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(input)
dense = MaxPooling2D(pool_size = (2, 2))(dense)
dense = BatchNormalization()(dense)
dense = Dropout(0.3)(dense)

flatten = Flatten()(dense)

dense = Dense(128, actiavation = 'relu')(flatten)
dense = 