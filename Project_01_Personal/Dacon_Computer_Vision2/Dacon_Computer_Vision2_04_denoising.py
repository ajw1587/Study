import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
filepath = '../data/csv/Computer_Vision2/train_dirty_mnist_2nd/00009.png'
img = cv.imread(filepath)
img2 = img.copy()
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img2 = cv.resize(img2, dsize = (128, 128))


# 임계처리
_, img2 = cv.threshold(img2, 173, 255, cv.THRESH_BINARY)

# 2차 노이즈 제거
kernel = np.ones((1, 1), np.uint8)
img2 = cv.dilate(img2, kernel, iterations = 1)
# img2 = cv.morphologyEx(img2, cv.MORPH_OPEN, np.ones((1, 1), np.uint8), iterations = 5)

# Blur 처리
# im2 = cv.GaussianBlur(img2, (5, 5), 0)
img2 = cv.medianBlur(img2, ksize = 3)
img2 = cv.medianBlur(img2, ksize = 3)

# kernel = np.ones((2, 2), np.uint8)
# _, img2 = cv.threshold(img2, 170, 255, cv.THRESH_BINARY)
# img2 = cv.dilate(img2, kernel)
# img2 = cv.morphologyEx(img2, cv.MORPH_CLOSE, kernel)
# img2 = cv.morphologyEx(img2, cv.MORPH_OPEN, kernel)
# img2 = cv.bilateralFilter(img2, 10, 50, 50)

plt.figure(figsize = (10, 5))
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
    subset = cv.resize(subset, dsize = (128, 128))

    # 임계처리
    _, img2 = cv.threshold(img2, 173, 255, cv.THRESH_BINARY)

    # 2차 노이즈 제거
    kernel = np.ones((1, 1), np.uint8)
    img2 = cv.dilate(img2, kernel, iterations = 1)
    # img2 = cv.morphologyEx(img2, cv.MORPH_OPEN, np.ones((1, 1), np.uint8), iterations = 5)

    # Blur 처리
    # im2 = cv.GaussianBlur(img2, (5, 5), 0)
    img2 = cv.medianBlur(img2, ksize = 3)
    img2 = cv.medianBlur(img2, ksize = 3)

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
    subset = cv.resize(subset, dsize = (128, 128))

    # 임계처리
    _, img2 = cv.threshold(img2, 173, 255, cv.THRESH_BINARY)
    
    # 2차 노이즈 제거
    kernel = np.ones((1, 1), np.uint8)
    img2 = cv.dilate(img2, kernel, iterations = 1)
    # img2 = cv.morphologyEx(img2, cv.MORPH_OPEN, np.ones((1, 1), np.uint8), iterations = 5)
    
    # Blur 처리
    # im2 = cv.GaussianBlur(img2, (5, 5), 0)
    img2 = cv.medianBlur(img2, ksize = 3)
    img2 = cv.medianBlur(img2, ksize = 3)

    # resize
    # subset = cv2.resize(subset, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)

    # test_img 리스트에 추가
    test_img.append(subset)

# list -> numpy 변환
train_img = np.array(train_img)
test_img = np.array(test_img)

train_img = train_img/255.
test_img = test_img/255.

# numpy 저장
np.save('../data/csv/Computer_Vision2/numpy_file/Train_Computer_Vision2_denoising128.npy',
        train_img)
np.save('../data/csv/Computer_Vision2/numpy_file/Test_Computer_Vision2_denoising128.npy',
        test_img)

'''
############################################################################################
# load data
x_train = np.load('../data/csv/Computer_Vision2/numpy_file/Train_Computer_Vision2_denoising128.npy')
x_test = np.load('../data/csv/Computer_Vision2/numpy_file/Test_Computer_Vision2_denoising128.npy')
y_train = pd.read_csv('../data/csv/Computer_Vision2/train_dirty_mnist_2nd_answer.csv')
y_submission = pd.read_csv('../data/csv/Computer_Vision2/sample_submission.csv')

x_train = x_train.reshape(50000, 128, 128, 1)
x_test = x_test.reshape(5000, 128, 128, 1)

# train_split_data
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

print(x_train.shape)        # (50000, 128, 128, 1)
print(x_test.shape)         # (5000, 128, 128, 1)
print(y_train.shape)        # (50000, 26)
print(y_submission.shape)   # (5000, 26)
print(type(x_train))        # <class 'numpy.ndarray'>
print(type(y_train))        # <class 'numpy.ndarray'>

# model
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Input, Activation, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
opt = Adam(learning_rate = 0.001)

def my_model():
    input1 = Input(shape = (128, 128, 1))
    x = Conv2D(32, (3, 3), padding = 'same')(input1)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3,3))(x)
    x = Dropout(0.25)(x)
    
    # x = Conv2D(64, (3, 3), padding = 'same')(input1)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding = 'same')(input1)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Dropout(0.25)(x)

    # x = Conv2D(128, (3, 3), padding = 'same')(input1)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(128, (3, 3), padding = 'same')(input1)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = MaxPooling2D(pool_size = (2,2))(x)
    # x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(1)(x)
    output1 = Activation('sigmoid')(x)
    model = Model(inputs = input1, outputs = output1)

    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return model



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 300, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 150, mode = 'auto')

model = my_model()

from string import ascii_lowercase
alpha = ascii_lowercase     # 알파벳 리스트
for i in alpha:
    cp_path = '../data/modelcheckpoint/Computer_Vision2/Dacon_Computer_Vision2_denoising128_' + str(i) + '.hdf5'
    cp = ModelCheckpoint(cp_path, save_best_only = True, mode = 'auto')

    # y_train, y_val에서 필요한 alphabet 추출
    y_train_alpha = y_train.loc[:, i].values
    y_val_alpha = y_val.loc[:, i].values

    hist = model.fit(x_train, y_train_alpha, epochs = 1000, validation_data = (x_val, y_val_alpha),
                     callbacks = [es, cp, reduce_lr])

    # 6. Predict
    model2 = load_model(cp_path)
    y_pred = model2.predict(x_test)
    y_pred = np.where(y_pred < 0.5, 0, 1)
    print(y_pred)
    print(y_pred.shape)
    y_submission.loc[:, i] = y_pred
    y_submission.to_csv('../data/modelcheckpoint/Computer_Vision2/Computer_Vision2_submission_denoising128.csv', index = False)