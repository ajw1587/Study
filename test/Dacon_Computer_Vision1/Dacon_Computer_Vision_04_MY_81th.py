import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold # KFold보다 데이터셋의 대표성을 더 잘 나타내준다.
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# File Path
train_filepath = '../data/csv/Computer_Vision/data/train.csv'
test_filepath = '../data/csv/Computer_Vision/data/test.csv'
submission_filepath = '../data/csv/Computer_Vision/data/submission.csv'
check_filepath = '../data/modelcheckpoint/Computer_Vision/Computer_Vision_best_model.h5'

# 1. 데이터 불러오기
train_dataset = pd.read_csv(train_filepath, engine = 'python', encoding = 'CP949')
test_dataset = pd.read_csv(test_filepath, engine = 'python', encoding = 'CP949')
submission_dataset = pd.read_csv(submission_filepath, engine = 'python', encoding = 'CP949')

# 2. 데이터 필요 column 추려주기
train_x = train_dataset.copy().drop(['id', 'letter', 'digit'], 1).values
train_y = train_dataset.copy().loc[:,'digit'].values
test_x = test_dataset.copy().drop(['id', 'letter'], 1).values
submission_data = submission_dataset.copy()

train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)

print(train_x.shape)            # (2048, 784)
print(train_y.shape)            # (2048,)
print(test_x.shape)             # (20480, 784)
print(submission_data.shape)    # (20480, 2)
# test 값이 train보다 압도적으로 많기 때문에 Augmentation 필요!

# 데이터 값 확인하기
# plt.imshow(train_x[0].reshape(28, 28))
# plt.show()

# 3. Image Augmentation
train_aug = ImageDataGenerator(rescale = 1./255., height_shift_range=(-1,1), width_shift_range=(-1,1))
# rotation_range = 10, shear_range = 0.2 이미지 변형이 일어나서 그런지 acc값이 더 좋지 않다.
# 옵션을 많이 넣으면 loss가 떨어지질 않는다!
# valid에도 옵션 추가함! 그런데 의미가 있나??? 의미 없음
valid_aug = ImageDataGenerator(rescale = 1./255.)
test_aug = ImageDataGenerator(rescale = 1./255.)

# 4. 모델 및 Fit
reduce_lr = ReduceLROnPlateau(factor = 0.5, patience = 60, mode = 'auto')
es = EarlyStopping(monitor = 'loss', patience = 120, mode = 'auto')
cp = ModelCheckpoint(check_filepath, monitor = 'loss', save_best_only=True, mode = 'auto')
opti = Adam(learning_rate = 0.001, epsilon = None)
skf = StratifiedKFold(n_splits = 20, random_state = 77)    # random_state 사용을 왜 안했니! 멍청아!
result = 0
Acc = []
i = 1

for train_index, valid_index in skf.split(train_x, train_y):
    print(i, '번째 실행 시작')

    x_train = train_x[train_index]
    x_valid = train_x[valid_index]
    y_train = train_y[train_index]
    y_valid = train_y[valid_index]

    train_generator = train_aug.flow(x_train, y_train)
    valid_generator = valid_aug.flow(x_valid, y_valid)
    x_test_generator = test_aug.flow(test_x, shuffle = False)
    # print(x_train_generator)
    # print(type(x_train_generator))
    # print(x_train_generator.dtype)

    input1 = Input(shape = (28, 28, 1))
    dense1 = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(input1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(32, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(16, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(8, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = MaxPooling2D((2, 2))(dense1)
    dense1 = Dropout(0.3)(dense1)

    dense1 = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(32, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(16, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(8, 2, 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = MaxPooling2D((2, 2))(dense1)
    dense1 = Dropout(0.3)(dense1)

    dense1 = Flatten()(dense1)
    dense1 = Dense(64, activation = 'relu')(dense1)
    dense1 = Dropout(0.3)(dense1)
    dense1 = Dense(32, activation = 'relu')(dense1)
    dense1 = Dense(16, activation = 'relu')(dense1)
    output1 = Dense(10, activation = 'softmax')(dense1)
    model = Model(inputs = input1, outputs = output1)

    # Compile and Fit
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opti, metrics = ['acc'])
    hist = model.fit_generator(train_generator, epochs = 2000, 
                               validation_data = valid_generator,
                               callbacks = [reduce_lr, es, cp])
                               # ../data/modelcheckpoint/Conputer_Vision/Computer_Vision_best_mode.h5
    hist = pd.DataFrame(hist.history)
    Acc.append(hist['val_acc'].max())

    # Predict
    model.load_weights(filepath = check_filepath)
    y_pred = model.predict_generator(x_test_generator)
    print(type(y_pred))
    result += y_pred

    submission_data['digit'] = np.argmax(result, axis = 1)
    submission_data.to_csv('../data/modelcheckpoint/Computer_Vision/Dacon_submission_data.csv', 
                           index = False)
    print(i, '번째 실행 끝')
    i += 1

print('ACC: ', np.mean(Acc))

# 0.93137, 81등