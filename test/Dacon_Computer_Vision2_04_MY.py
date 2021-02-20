import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
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

y_train = y_train.iloc[:, 1:].values

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
# print(y_submission.columns)
# Index(['index', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
#        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
#       dtype='object')

# 4. 모델
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, Flatten
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def my_model(drop = 0.5, size = 64):
    input1 = Input(shape = (size, size, 3))
    dense1 = Conv2D(128, 3, 1, padding = 'same', activation = 'relu')(input1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Conv2D(64, 3, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Conv2D(32, 3, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Conv2D(16, 3, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)

    dense1 = Conv2D(128, 3, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Conv2D(64, 3, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Conv2D(32, 3, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)

    dense1 = Flatten()(dense1)
    dense1 = Dense(32, activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Dense(16, activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    output1 = Dense(26, activation = 'softmax')(dense1)

    model = Model(inputs = input1, outputs = output1)
    model.compile(optimizer = SGD(lr=0.001, momentum=0.90, decay=1, nesterov=False), loss = 'categorical_crossentropy', metrics = ['acc'])
    return model
model = my_model()

# 5. Fit
es = EarlyStopping(monitor = 'val_loss', patience = 100, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 50, mode = 'auto')
cp_path = '../data/modelcheckpoint/Computer_Vision2/model/Dacon_Computer_Vision2_04.hdf5'
cp = ModelCheckpoint(cp_path, save_best_only = True, mode = 'auto')

model.fit(x_train, y_train, batch_size = 128, epochs = 1000, callbacks = [es, cp, reduce_lr],
          validation_data = (x_val, y_val))

# 6. Predict
model2 = load_model(cp_path)
y_pred = model2.predict(x_test)
y_pred = np.where(y_pred < 0.5, 0, 1)
print(y_pred)
print(y_pred.shape)
y_submission.iloc[:, 1:] = y_pred
y_submission.to_csv('../data/modelcheckpoint/Computer_Vision2/submission/submission_04.csv', index = False)

# val_acc가 acc보다 잘나온다면 Dropout을 건드려보자!(낮추거나 없애거나)
# Dropout은 train에만 적용이 되어 val_acc가 acc보다 높을 수 있다!