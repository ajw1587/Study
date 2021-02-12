import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

    # train_img 리스트에 추가
    test_img.append(subset)

# list -> numpy 변환
train_img = np.array(train_img)
test_img = np.array(test_img)

# numpy 저장
np.save('../data/csv/Computer_Vision2/numpy_file/Train_Computer_Vision2_02.npy',
        train_img)
np.save('../data/csv/Computer_Vision2/numpy_file/Test_Computer_Vision2_02.npy',
        test_img)

# 이미지가 너무 많아서 오래걸림...
# 굳이 5만개를 다 할 필요가 있을까 모르겠네...
'''

# 2. 노이즈 제거 데이터 불러오기
x_train = np.load('../data/csv/Computer_Vision2/numpy_file/Train_Computer_Vision2_02.npy')
x_test = np.load('../data/csv/Computer_Vision2/numpy_file/Test_Computer_Vision2_02.npy')
y_train = pd.read_csv('../data/csv/Computer_Vision2/train_dirty_mnist_2nd_answer.csv')
y_submission = pd.read_csv('../data/csv/Computer_Vision2/sample_submission.csv')
y_train = y_train.values
y_submission = y_submission.values

# (50000, 256, 256, 3)
# (5000, 256, 256, 3)
# (50000, 27)
# (5000, 27)

# 3. 전처리
# x_train = x_train/255.
# x_test = x_test/255.
# ERROR 발생
# MemoryError: Unable to allocate 73.2 GiB for an array with shape (50000, 256, 256, 3) and data type float64

# 4. 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, KFold
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def my_model(drop = 0.3, size = 256):
    input1 = Input(shape = (size, size, 3))
    dense1 = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(input1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(32, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(16, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(8, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = MaxPooling2D((2, 2))(dense1)
    dense1 = Dropout(drop)(dense1)

    # dense1 = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Conv2D(32, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Conv2D(16, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Conv2D(8, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = MaxPooling2D((2, 2))(dense1)
    # dense1 = Dropout(drop)(dense1)

    dense1 = Flatten()(dense1)
    dense1 = Dense(64, activation = 'relu')(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Dense(32, activation = 'relu')(dense1)
    dense1 = Dense(16, activation = 'relu')(dense1)
    output1 = Dense(1, activation = 'sigmoid')(dense1)
    model = Model(inputs = input1, outputs = output1)
    model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'binary_crossentropy', metrics = ['acc'])
    return model
# def my_hyper():
#     drop = [0.3]
#     size = [256]
#     epochs = [1]
#     return {'drop': drop,
#             'size': size,
#             'epochs': epochs}

# model = KerasClassifier(build_fn = my_model, verbose = 1)
# my_hyper = my_hyper()
# kfold = KFold(n_splits = 5, random_state = 77)
# kfold = StratifiedKFold(n_splits = 20, random_state = 77)
# n_splits = 5, 10 ERROR 발생
# ValueError: n_splits=10 cannot be greater than the number of members in each class.
# search = RandomizedSearchCV(model, my_hyper, cv = kfold)

model = my_model()
es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto')
cp = ModelCheckpoint('../data/modelcheckpoint/Computer_Vision2/Dacon_Computer_Vision2.hdf5')
for i in range(1): # 28
    model.fit(x_train, y_train[:, i], batch_size = 64, callbacks = [es, cp])