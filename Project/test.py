import cv2
import numpy as np
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 
# 22, 'x': 23, 'y': 24, 'z': 25}

def empty(a):
    pass

data_save_path = '../data/sign_image/sign_language01/numpy_data'

# Load Data
x_train = np.load(data_save_path + '/sign_language01_color_x.npy')
y_train = np.load(data_save_path + '/sign_language01_y.npy')

print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
# x_train.shape:  (39000, 64, 64, 3)
# y_train.shape:  (39000, 26)


cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars', 640, 240)
# MinMax로 나누는 이뉴는 Min과 Max 사이에 있는 색범위를 추출하기 위해서다.
# cv2.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
# cv2.createTrackbar('Hue Max', 'TrackBars', 33, 179, empty)
# cv2.createTrackbar('Sat Min', 'TrackBars', 63, 255, empty)
# cv2.createTrackbar('Sat Max', 'TrackBars', 255, 255, empty)
# cv2.createTrackbar('Val Min', 'TrackBars', 89, 255, empty)
# cv2.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)

img1 = cv2.cvtColor(x_train[100], cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(x_train[200], cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(x_train[300], cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(x_train[400], cv2.COLOR_BGR2GRAY)
img5 = cv2.cvtColor(x_train[500], cv2.COLOR_BGR2GRAY)

img1 = cv2.resize(img1, dsize = (480, 480), interpolation = cv2.INTER_LINEAR)
img2 = cv2.resize(img2, dsize = (480, 480), interpolation = cv2.INTER_LINEAR)
img3 = cv2.resize(img3, dsize = (480, 480), interpolation = cv2.INTER_LINEAR)
img4 = cv2.resize(img4, dsize = (480, 480), interpolation = cv2.INTER_LINEAR)
img5 = cv2.resize(img5, dsize = (480, 480), interpolation = cv2.INTER_LINEAR)

img1 = img1.astype(np.uint8)
img2 = img2.astype(np.uint8)
img3 = img3.astype(np.uint8)
img4 = img4.astype(np.uint8)
img5 = img5.astype(np.uint8)

img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
img3 = cv2.adaptiveThreshold(img3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
img4 = cv2.adaptiveThreshold(img4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
img5 = cv2.adaptiveThreshold(img5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

print(type(img1))

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)
cv2.imshow('img5', img5)

    # print(img.shape)
    # imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# h_min = 0
# h_max = 33
# s_min = 63
# s_max = 255
# v_min = 89
# v_max = 255
'''
while True:
    img1 = cv2.cvtColor(x_train[100], cv2.COLOR_BGR2HSV)
    img2 = cv2.cvtColor(x_train[200], cv2.COLOR_BGR2HSV)
    img3 = cv2.cvtColor(x_train[300], cv2.COLOR_BGR2HSV)
    img4 = cv2.cvtColor(x_train[400], cv2.COLOR_BGR2HSV)
    img5 = cv2.cvtColor(x_train[500], cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')      # 0
    h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')      # 33
    s_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')      # 63
    s_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')      # 255
    v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')      # 89
    v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')      # 255

    print(h_min, h_max, s_min, s_max, v_min, v_max)

    # #     # 적용될 색 영역 설정
    # # print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    img1 = cv2.inRange(img1, lower, upper)
    img2 = cv2.inRange(img2, lower, upper)
    img3 = cv2.inRange(img3, lower, upper)
    img4 = cv2.inRange(img4, lower, upper)
    img5 = cv2.inRange(img5, lower, upper)

    img1 = cv2.resize(img1, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
    img3 = cv2.resize(img3, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
    img4 = cv2.resize(img4, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
    img5 = cv2.resize(img5, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('img3', img3)
    cv2.imshow('img4', img4)
    cv2.imshow('img5', img5)

    print(h_min, h_max, s_min, s_max, v_min, v_max)

    if cv2.waitKey(20) == 27:
        break
'''

    

cv2.waitKey()

cv2.destroyAllWindows()