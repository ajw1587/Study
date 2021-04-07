import numpy as np
import tensorflow as tf
import cv2 as cv
import glob
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Flatten

# Data
img = glob.glob('F:/Team Project/OCR/Image_to_Text_model/image-data/hangul-images/*.jpeg')
print(len(img))
# 37600ea Image Data
# img = cv.imread(img[0], cv.IMREAD_GRAYSCALE)
# img = cv.resize(img, dsize = (128, 128))
# cv.imshow('img', img)
# cv.moveWindow('img', 300, 300)
# cv.waitKey(0)
# cv.destroyAllWindows()
for i in range(1, len(img) + 1):
    



# input = Input(shape = (128, 128, 1))
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

# dense = Dense(128, actiavation = 'relu')(flatten)
# dense = 