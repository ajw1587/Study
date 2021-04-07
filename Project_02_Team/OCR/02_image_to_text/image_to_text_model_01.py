import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization

img = cv.imread('F:/Team Project/OCR/Text_detection/image_data/test_picture/test2/test0_0.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, dsize = (128, 128))
# cv.imshow('img', img)
# cv.moveWindow('img', 300, 300)
# cv.waitKey(0)
# cv.destroyAllWindows()

img = img.reshape(img.shape[0], img.shape[1], 1)
print(img.shape)
