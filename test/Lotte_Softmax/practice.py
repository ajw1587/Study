# 보간법 비교인데...
# 별 차이점을 못느끼겠다...

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# image01 = cv.imread('../data/lotte/train/0/0.jpg')
# image01 = cv.resize(image01, (100, 100), interpolation = cv.INTER_CUBIC)

# image02 = cv.imread('../data/lotte/train/0/0.jpg')
# image02 = cv.resize(image02, (100, 100), interpolation = cv.INTER_AREA)

# image03 = cv.imread('../data/lotte/train/0/0.jpg')
# image03 = cv.resize(image03, (100, 100), interpolation = cv.INTER_LINEAR)

# image04 = cv.imread('../data/lotte/train/0/0.jpg')
# image04 = cv.resize(image04, (100, 100), interpolation = cv.INTER_NEAREST)

# plt.subplot(2, 2, 1)
# plt.imshow(image01)
# plt.subplot(2, 2, 2)
# plt.imshow(image02)
# plt.subplot(2, 2, 3)
# plt.imshow(image03)
# plt.subplot(2, 2, 4)
# plt.imshow(image04)

# plt.show()

x_train = np.load('../data/lotte/train_x(256,256).npy')
y_train = np.load('../data/lotte/train_y(256,256).npy')
x_predict = np.load('../data/lotte/predict_x(256,256).npy')

print(x_train.shape)
print(y_train.shape)
print(x_predict.shape)