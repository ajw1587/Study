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
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

# 3. Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./254,
    height_shift_range = 0.2,
    width_shift_range = 0.2
)

val_datagen = ImagedataGenerator(
    rescale = 1./254
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

# 4. 모델
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Input, Flatten
from tensorflow.keras.optimizers import Adam, Adadelta, SGD
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
    # dense1 = Conv2D(16, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Dropout(drop)(dense1)
    # dense1 = Conv2D(8, 2, 1, padding = 'same', activation = 'relu')(dense1)
    # dense1 = MaxPooling2D((2, 2))(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Dropout(drop)(dense1)

    dense1 = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
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
    # dense1 = Dense(64, activation = 'relu')(dense1)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Dropout(drop)(dense1)
    dense1 = Dense(32, activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(drop)(dense1)
    dense1 = Dense(16, activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    output1 = Dense(1, activation = 'sigmoid')(dense1)

    model = Model(inputs = input1, outputs = output1)
    model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['acc'])
    return model
model = my_model()

# 5. Fit
from string import ascii_lowercase

es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 25, mode = 'auto')
alpha = ascii_lowercase     # 알파벳 리스트

for i in alpha:
    cp_path = '../data/modelcheckpoint/Computer_Vision2/model/Dacon_Computer_Vision2_Aug_' + str(i) + '.hdf5'
    cp = ModelCheckpoint(cp_path, save_best_only = True, mode = 'auto')

    # y_train, y_val에서 필요한 alphabet 추출
    y_train_alpha = y_train.loc[:, i].values
    y_val_alpha = y_val.loc[:, i].values

    # Generator 적용
    train_flow = train_datagen.flow(x_train, y_train_alpha, batch_size = 128, seed = 77, class_mode = 'binary')
    val_flow = val_datagen.flow(x_val, y_val_alpha)
    test_flow = test_datagen.flow(x_test)

    model.fit_generator(train_flow,
                        steps_per_epoch = 40000//128,
                        epochs = 100,
                        callbacks = [es, cp, reduce_lr],
                        validation_data = val_flow,
                        validation_steps = 10000//128)

    # 6. Predict
    model2 = load_model(cp_path)
    y_pred = model2.predict(x_test)
    y_pred = np.where(y_pred < 0.5, 0, 1)
    print(y_pred)
    print(y_pred.shape)
    y_submission.loc[:, i] = y_pred
y_submission.to_csv('../data/modelcheckpoint/Computer_Vision2/submission/submission_Aug.csv', index = False)