import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
'''
# 1. 데이터

# file name: 'hand(1. 숫자)_(2. 숫자)_bot_seg_(3. 숫자)_cropped.jpeg'
# 메모리 사용 줄이기 위해 Image_Size = 400 x 400 -> 64 x 64 변경 예정

# 손상된 이미지 걸러주기 -> 그렇게 자주 사용하지는 않는 기능이다.
# 헤더에 'JFIF' 문자열이 없는 잘못된 이미지를 삭제.
# https://stackoverflow.com/questions/62220855/tensorflow-removing-jfif

# Data 경로
file_path = '../data/sign_image/sign_language/asl_dataset/asl_dataset'

# ImageDataGenerator 옵션 설정
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 0.3
)
except_datagen = ImageDataGenerator(
    rescale = 1./255
)

# ImageDataGenerator Flow 설정
batchsize = 64
targetsize = 64
train_flow = train_datagen.flow_from_directory(
    file_path, target_size = (targetsize, targetsize), class_mode = 'categorical',
    batch_size = batchsize, seed = 77)
# Found 2515 images belonging to 36 classes.

valid_flow = train_datagen.flow_from_directory(
    file_path, target_size = (targetsize, targetsize), class_mode = 'categorical',
    batch_size = batchsize, seed = 70)

test_flow = except_datagen.flow_from_directory(
    file_path, target_size = (targetsize, targetsize), class_mode = 'categorical',
    batch_size = batchsize, seed = 65)

print(train_flow[0][0].shape)
print(train_flow[0][1].shape)
print(type(train_flow))
print(train_flow.class_indices)
print(type(train_flow[0][0][0]))
print(type(train_flow[0][0]))
print(type(train_flow[0]))

# '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
# 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 
# 'j': 19, 'k': 20, 'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27, 
# 's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34, 'z': 35

plt.imshow(train_flow[0][0][0])
plt.show()


# 2. 모델
opti = Adam(learning_rate = 0.001)
es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto')
cp = ModelCheckpoint('../data/modelcheckpoint/sign_language/sign_language_model_02.hdf5')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, mode = 'auto')
def my_model():
    input1 = Input(shape = (64, 64, 3))
    x = Conv2D(128, (5, 5), 1, padding = 'same', activation = 'relu')(input1)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (5, 5), 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, (5, 5), 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (3, 3))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(32, activation = 'relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation = 'relu')(x)
    output1 = Dense(36, activation = 'softmax')(x)
    model = Model(inputs = input1, outputs = output1)

    model.compile(loss = 'categorical_crossentropy', optimizer = opti, metrics = ['acc'])

    return model

model = my_model()
hist = model.fit_generator(train_flow,
                           steps_per_epoch = 2515//64,
                           epochs = 200,
                           validation_data = valid_flow,
                           validation_steps = 10,
                           callbacks = [es, reduce_lr])
acc = hist.history['acc']
loss = hist.history['loss']
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']

print('acc: ', acc[-1])
print('loss: ', loss[-1])
print('val_acc: ', val_acc[-1])
print('val_loss: ', val_loss[-1])

# acc:  0.9314565658569336
# loss:  0.17483144998550415
# val_acc:  0.9828125238418579
# val_loss:  0.04662995785474777
'''

# x_pred01 = Image.open('../data/sign_image/one.png')
# x_pred02 = Image.open('../data/sign_image/five.png')

x_pred01 = cv2.imread('../data/sign_image/one.png')
x_pred02 = cv2.imread('../data/sign_image/five.png')

# print(x_pred01.shape)       # (250, 265, 3)
# print(x_pred02.shape)       # (264, 394, 3)
# print(type(x_pred01))       # <class 'numpy.ndarray'>
# print(type(x_pred02))       # <class 'numpy.ndarray'>

x_pred01 = cv2.resize(x_pred01, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
x_pred02 = cv2.resize(x_pred02, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)

# x_pred01 = np.array(x_pred01)
# x_pred02 = np.array(x_pred02)

x_pred01 = x_pred01.reshape(1, 64, 64, 3)
x_pred02 = x_pred02.reshape(1, 64, 64, 3)

model2 = load_model('../data/modelcheckpoint/sign_language/sign_language_model_02.hdf5')
pred01 = model2.predict(x_pred01)
pred02 = model2.predict(x_pred02)

pred01 = np.argmax(pred01, axis = 1)
pred02 = np.argmax(pred02, axis = 1)

print(pred01)
print(pred02)
