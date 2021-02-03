# Computer_Vision
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. Train 데이터 y -> 0~9, 다중분류
file_path = '../data/csv/Computer_Vision/data/train.csv'
test_file_path = '../data/csv/Computer_Vision/data/test.csv'
dataset = pd.read_csv(file_path, engine = 'python', encoding = 'CP949')
test_dataset = pd.read_csv(test_file_path, engine = 'python', encoding = 'CP949')

x = dataset.drop(['id', 'letter', 'digit'], axis = 1).values
y = dataset.iloc[:, 1].values
test = test_dataset.drop(['id', 'letter'], axis = 1).values
# print(x.shape)      (2048, 784)
# print(y.shape)      (2048,)
# print(test.shape)   (20480, 784)
# print(type(x))      <class 'numpy.ndarray'>
# print(type(y))      <class 'numpy.ndarray'>
# print(type(test))   <class 'numpy.ndarray'>

# 그래프로 살펴보기
import matplotlib.pyplot as plt
import seaborn as sns

# plt.figure(figsize = (10, 6))
# sns.set_theme(style = 'darkgrid')
# ax = sns.countplot(x = 'digit', data = dataset)
# ax.set(xlabel = 'Digit(Target)', ylabel = 'Count')
# plt.show()

# plt.figure(figsize = (10, 6))
# sns.set_theme(style = 'darkgrid')
# ax = sns.countplot(x = 'letter', data = dataset)
# ax.set(xlabel = 'Letter(Target)', ylabel = 'Count')
# plt.show()

x = x.reshape(-1, 28, 28, 1)
test = test.reshape(-1, 28, 28, 1)

x = x/255.
test = test/255.
# ImageDataGenerator & data augmentation
# idg = ImageDataGenerator(rescale=1./255,           # 리스케일링
#                          rotation_range = 20,      # 이미지 회전
#                          width_shift_range=0.3,    # 좌우 이동
#                          height_shift_range=0.3,   # 상하 이동
#                          shear_range=0.5,          # 밀림 강도
#                          zoom_range=0.2,           # 확대
#                          horizontal_flip=True,     # 좌우 반전
#                          vertical_flip=True)       # 상하 반전
# https://tykimos.github.io/2017/06/10/CNN_Data_Augmentation/
idg = ImageDataGenerator(rotation_range = 20, height_shift_range=(-1,1), width_shift_range=(-1,1))
idg2 = ImageDataGenerator()

# # show augmented image data
# sample_data = x[100].copy()
# sample = sample_data.reshape(1, 28, 28, 1)
# # sample = np.expand_dims(sample_data,0)
# sample_datagen = ImageDataGenerator(height_shift_range=(-1,1), width_shift_range=(-1,1))
# sample_generator = sample_datagen.flow(sample, batch_size=1)

# plt.figure(figsize=(16,10))
# for i in range(9) : 
#     plt.subplot(3,3,i+1)
#     # sample_batch = sample_generator.next()
#     # sample_image=sample_batch[0]
#     plt.imshow(sample_generator.next()[0].reshape(28,28))
# plt.show()

# cross validation
kfold = StratifiedKFold(n_splits = 20, random_state = 77, shuffle = True)
# kfold = KFold(n_splits = 2, random_state = 77, shuffle = True)
es = EarlyStopping(patience = 100, mode = 'auto')
reduce_lr = ReduceLROnPlateau(factor = 0.5, patience = 50, mode = 'auto')
opti = Adam(learning_rate = 0.0001)     # , epsilon = None
submission_path = '../data/csv/Computer_Vision/data/submission.csv'

ACC = [] # loss = []
result = 0
i = 1
for train_index, valid_index in kfold.split(x, y):

    print(i, '번째 실행시작')

    # KFold: n_splits = 40, for문 40번 반복
    # 즉, 이미지 Augmentation을 40번 반복, ex) (기존이미지 4장 -> 8장)을 40번 반복
    x_train = x[train_index]
    x_valid = x[valid_index]
    y_train = y[train_index]
    y_valid = y[valid_index]

    train_generator = idg.flow(x_train, y_train, batch_size = 8)
    valid_generator = idg2.flow(x_valid, y_valid)
    test_generator = idg2.flow(test, shuffle = False)

    # model = Sequential()
    
    # model.add(Conv2D(16,(3,3),activation='relu',input_shape=(28,28,1),padding='same'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(32,(5,5),activation='relu',padding='same')) 
    # model.add(BatchNormalization())
    # model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((3,3)))
    # model.add(Dropout(0.3))
    
    # model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64,(5,5),activation='relu',padding='same')) 
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((3,3)))
    # model.add(Dropout(0.3))
    
    # model.add(Flatten())

    # model.add(Dense(128,activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    # model.add(Dense(64,activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    # model.add(Dense(10,activation='softmax'))
    input1 = Input(shape = (28, 28, 1))
    conv2d = Conv2D(64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(input1)
    dense1 = Conv2D(64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(conv2d)
    dense1 = MaxPooling2D(pool_size = (2, 2))(dense1)
    dense1 = Dropout(0.3)(dense1)

    dense1 = Conv2D(64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = Conv2D(64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = Conv2D(32, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = MaxPooling2D(pool_size = (2, 2))(dense1)
    dense1 = Dropout(0.3)(dense1)

    dense1 = Conv2D(64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = Conv2D(64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = Conv2D(32, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
    dense1 = MaxPooling2D(pool_size = (2, 2))(dense1)
    dense1 = Dropout(0.3)(dense1)

    dense1 = Flatten()(dense1)
    dense1 = Dense(128, activation = 'relu')(dense1)
    dense1 = Dropout(0.3)(dense1)
    dense1 = Dense(64, activation = 'relu')(dense1)
    dense1 = Dense(32, activation = 'relu')(dense1)
    dense1 = Dense(16, activation = 'relu')(dense1)
    output1 = Dense(10, activation = 'softmax')(dense1)
    model = Model(inputs = input1, outputs = output1)

    cp_path = '../data/modelcheckpoint/Computer_Vision/Computer_Vision_best_model.h5'
    cp = ModelCheckpoint(cp_path, save_best_only = True, mode = 'auto')
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opti, metrics = ['acc'])
    # sparse_categorical_crossentropy: onehotencoding 적용하지 않고 사용하는 loss
    hist = model.fit_generator(train_generator, epochs = 2000, validation_data = valid_generator,
                               callbacks = [es, reduce_lr, cp])
    categori_loss, acc = model.evaluate_generator(test_generator)

    hist = pd.DataFrame(hist.history)
    ACC.append(hist['val_acc'].max())
    # model.save(cp_path)         # weight 저장

    # Predict
    model.load_weights(filepath = cp_path)
    result += model.predict_generator(test_generator)/40

    submission_data = pd.read_csv(submission_path, encoding = 'CP949', engine = 'python')
    submission_data['digit'] = result.argmax(1)
    print(submission_data.head())
    submission_data.to_csv('../data/modelcheckpoint/Computer_Vision/Dacon_submission_data.csv', index = False)

    print(i, '번째 실행끝')
    i += 1


finally_loss = np.mean(ACC)
print('finally_loss', finally_loss)