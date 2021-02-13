import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 이미지가 너무 많아서 오래걸림...
# 굳이 5만개를 다 할 필요가 있을까 모르겠네...
# resize를 하자
'''
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
    subset = cv2.imread(train_path)
    
    # Noise 제거
    subset = cv2.fastNlMeansDenoisingColored(subset,None,20,20,7,21)

    # resize (50000,256,256) 너무 크다 줄이자!
    subset = cv2.resize(subset, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)

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
    subset = cv2.imread(test_path)
    
    # Noise 제거
    subset = cv2.fastNlMeansDenoisingColored(subset,None,20,20,7,21)

    # resize
    subset = cv2.resize(subset, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)

    # test_img 리스트에 추가
    test_img.append(subset)

# list -> numpy 변환
train_img = np.array(train_img)
test_img = np.array(test_img)

# numpy 저장
np.save('../data/csv/Computer_Vision2/numpy_file/Train_Computer_Vision2_02.npy',
        train_img)
np.save('../data/csv/Computer_Vision2/numpy_file/Test_Computer_Vision2_02.npy',
        test_img)
'''


# 2. 노이즈 제거 데이터 불러오기
x_train = np.load('../data/csv/Computer_Vision2/numpy_file/Train_Computer_Vision2_02.npy')
x_test = np.load('../data/csv/Computer_Vision2/numpy_file/Test_Computer_Vision2_02.npy')
y_train = pd.read_csv('../data/csv/Computer_Vision2/train_dirty_mnist_2nd_answer.csv')
y_submission = pd.read_csv('../data/csv/Computer_Vision2/sample_submission.csv')

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_submission.shape)
# (50000, 64, 64, 3)
# (5000, 64, 64, 3)
# (50000, 27)
# (5000, 27)
# print(np.max(x_train))    # 254
# print(np.max(x_test))     # 254

# 3. 전처리
x_train = x_train/254.
x_test = x_test/254.
# (50000, 256, 256, 3) 상태에서 실행시 ERROR 발생
# MemoryError: Unable to allocate 73.2 GiB for an array with shape (50000, 256, 256, 3) and data type float64

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_submission.shape)
# print(y_train.loc[:, 'a'])

# 4. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def my_model(drop = 0.5, size = 64):
    input1 = Input(shape = (size, size, 3))
    dense1 = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(input1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Conv2D(32, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Conv2D(16, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Conv2D(8, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = MaxPooling2D((2, 2))(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)

    # dense1 = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Dropout(drop)(dense1)
    # dense1 = Conv2D(32, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Dropout(drop)(dense1)
    # dense1 = Conv2D(16, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Dropout(drop)(dense1)
    # dense1 = Conv2D(8, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = MaxPooling2D((2, 2))(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Dropout(drop)(dense1)

    dense1 = Flatten()(dense1)
    dense1 = Dense(64, activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Dense(32, activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Dense(16, activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    output1 = Dense(1, activation = 'sigmoid')(dense1)
    model = Model(inputs = input1, outputs = output1)
    model.compile(optimizer = Adam(learning_rate = 0.00001), loss = 'binary_crossentropy', metrics = ['acc'])
    return model
model = my_model()

# 5. Fit
from string import ascii_lowercase

es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, mode = 'auto')
alpha = ascii_lowercase     # 알파벳 리스트
for i in alpha:
    cp = ModelCheckpoint('../data/modelcheckpoint/Computer_Vision2/Dacon_Computer_Vision2_' + str(i) + '.hdf5',
                         save_best_only = True, mode = 'auto')

    y_alpha = y_train.loc[:, i].values
    model.fit(x_train, y_alpha, batch_size = 64, epochs = 100, callbacks = [es, cp, reduce_lr],
              validation_data = (x_val, y_val.loc[:, i].values))

# 6. Predict
for i in alpha:
    y_pred = model.predict(x_test)
    print(y_pred.shape)