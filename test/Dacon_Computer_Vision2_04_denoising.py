import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
filepath = '../data/csv/Computer_Vision2/train_dirty_mnist_2nd/00001.png'
img = cv.imread(filepath)
img2 = img.copy()
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img2 = cv.resize(img2, dsize = (256, 256))

# 2차 노이즈 제거
kernel = np.ones((2, 2), np.uint8)
img2 = cv.dilate(img2, kernel, iterations = 1)

# 임계처리
_, img2 = cv.threshold(img2, 254, 255, cv.THRESH_BINARY)

# Blur 처리
img2 = cv.medianBlur(img2, ksize = 3)

img2 = np.array(img2)
img2 = img2.reshape(256, 256, 1)

plt.figure(figsize = (20, 10))
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(img2)
plt.show()


##################################################################################
# 1. 데이터 노이즈 제거하여 가져오기
# 1-1. train_dirty_image
train_first_path = '../data/csv/Computer_Vision2/train_dirty_mnist_2nd/'
second_path = '.png'
train_img = []

for i in range(50000):
    # 경로 설정
    i = '{0:05d}'.format(i)
    train_path = train_first_path + i + second_path
    print(train_path)
    # 이미지 가져오기
    subset = cv.imread(train_path)
    
    # Noise 제거
    subset = cv.cvtColor(subset, cv.COLOR_BGR2GRAY)
    subset = cv.resize(subset, dsize = (256, 256))

    # 2차 노이즈 제거
    kernel = np.ones((2, 2), np.uint8)
    subset = cv.dilate(subset, kernel, iterations = 1)

    # 임계처리
    _, subset = cv.threshold(subset, 254, 255, cv.THRESH_BINARY)

    # Blur 처리
    subset = cv.medianBlur(subset, ksize = 3)

    # # resize (50000,256,256) 너무 크다 줄이자!
    ## subset = cv2.resize(subset, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
    # 크기를 너무 줄이면 무슨 알파벳인지 모른다!

    # train_img 리스트에 추가
    train_img.append(subset)

# 1-2. test_dirty_image
test_first_path = '../data/csv/Computer_Vision2/test_dirty_mnist_2nd/5'
second_path = '.png'
test_img = []

for i in range(5000):
    # 경로 설정
    i = '{0:04d}'.format(i)
    test_path = test_first_path + i + second_path
    print(test_path)
    # 이미지 가져오기
    subset = cv.imread(test_path)
    
    # Noise 제거
    subset = cv.cvtColor(subset, cv.COLOR_BGR2GRAY)
    subset = cv.resize(subset, dsize = (256, 256))

    # 2차 노이즈 제거
    kernel = np.ones((2, 2), np.uint8)
    subset = cv.dilate(subset, kernel, iterations = 1)

    # 임계처리
    _, subset = cv.threshold(subset, 254, 255, cv.THRESH_BINARY)

    # Blur 처리
    subset = cv.medianBlur(subset, ksize = 3)

    # resize
    # subset = cv2.resize(subset, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)

    # test_img 리스트에 추가
    test_img.append(subset)

# list -> numpy 변환
train_img = np.array(train_img)
test_img = np.array(test_img)

# numpy 저장
np.save('../data/csv/Computer_Vision2/numpy_file/Train_Computer_Vision2_denoising.npy',
        train_img)
np.save('../data/csv/Computer_Vision2/numpy_file/Test_Computer_Vision2_denoising.npy',
        test_img)
'''
############################################################################################
# load data
x_train = np.load('../data/csv/Computer_Vision2/numpy_file/Train_Computer_Vision2_denoising.npy')
x_test = np.load('../data/csv/Computer_Vision2/numpy_file/Test_Computer_Vision2_denoising.npy')
y_train = pd.read_csv('../data/csv/Computer_Vision2/train_dirty_mnist_2nd_answer.csv')
y_submission = pd.read_csv('../data/csv/Computer_Vision2/sample_submission.csv')

x_train = x_train.reshape(50000, 256, 256, 1)
x_test = x_test.reshape(5000, 256, 256, 1)
y_train = y_train.iloc[:, 1:].values
y_submission = y_submission.iloc[:, 1:].values

# train_split_data


# print(x_train.shape)        (50000, 256, 256, 1)
# print(x_test.shape)         (5000, 256, 256, 1)
# print(y_train.shape)        (50000, 26)
# print(y_submission.shape)   (5000, 26)
# print(type(x_train))        <class 'numpy.ndarray'>
# print(type(y_train))        <class 'numpy.ndarray'>

# model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Activation, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate = 1e-3)

def my_model():
    input1 = Input(shape = (256, 256, 1))
    x = Conv2D(32, (3, 3), padding = 'same')(input1)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), padding = 'same')(input1)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding = 'same')(input1)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding = 'same')(input1)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding = 'same')(input1)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(26)(x)
    output1 = Activation('softmax')(x)
    model = Model(inputs = input1, outputs = output1)

    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return model

model = my_model()
model.fit(x_train, y_train, epochs = 1, validate_)