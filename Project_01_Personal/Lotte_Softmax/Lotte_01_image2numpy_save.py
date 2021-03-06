import numpy as np
import cv2 as cv
import glob
from tensorflow.keras.utils import to_categorical

train_image_arr = []
train_y = []
print(train_y)
for i in range(1000):
    for j in range(48):
        path = '../data/lotte/train/' + str(i) + '/' + str(j) + '.jpg'
        print(path)
        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (192,192), interpolation = cv.INTER_CUBIC)
        
        train_image_arr.append(image)
        train_y.append(i)

train_image_arr = np.asarray(train_image_arr)
train_y = np.asarray(train_y)
train_y = train_y.reshape((-1, 1))
train_y = to_categorical(train_y)

print(train_image_arr.shape)
print(train_y)
print(train_y.shape)

np.save('../data/lotte/lotte_data/train_x(192,192).npy', arr = train_image_arr)
np.save('../data/lotte/lotte_data/train_y(192,192).npy', arr = train_y)

###############################################################

test_image_arr = []
for i in range(72000):
    path = '../data/lotte/test/' + str(i) + '.jpg'
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (192, 192), interpolation = cv.INTER_CUBIC)
    test_image_arr.append(image)
    print(i)
test_image_arr = np.asarray(test_image_arr)

np.save('../data/lotte/lotte_data/predict_x(192,192).npy', arr = test_image_arr)