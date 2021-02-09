# 실습
# ImageDataGenerator, fit 사용

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
'''
# 1. 데이터
train_aug = ImageDataGenerator(
    rescale = 1./255.,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
)

test_aug = ImageDataGenerator(
    rescale = 1./255.
)

train_xy = train_aug.flow_from_directory(
    '../data/image/gender/train',
    target_size = (30, 30),
    batch_size = 1900,
    class_mode = 'binary'
    # ,save_to_dir = '../data/image/gender_generator/train'
)

test_xy = test_aug.flow_from_directory(
    '../data/image/gender/train',
    target_size = (30, 30),
    batch_size = 300,
    class_mode = 'binary'
)


# npy로 바꿔주기
np.save('../data/image/gender/npy/keras67_2_train_x.npy', train_xy[0][0])
np.save('../data/image/gender/npy/keras67_2_train_y.npy', train_xy[0][1])
np.save('../data/image/gender/npy/keras67_2_test_x.npy', test_xy[0][0])
np.save('../data/image/gender/npy/keras67_2_test_y.npy', test_xy[0][1])
'''

# 1. 데이터 불러오기
x_train = np.load('../data/image/gender/npy/keras67_2_train_x.npy')
y_train = np.load('../data/image/gender/npy/keras67_2_train_y.npy')
x_test = np.load('../data/image/gender/npy/keras67_2_test_x.npy')
y_test = np.load('../data/image/gender/npy/keras67_2_test_y.npy')

print(x_train.shape)        # (1736, 30, 30, 3)
print(y_train.shape)        # (1736,)
print(x_test.shape)         # (300, 30, 30, 3)
print(y_test.shape)         # (300,)

# 2. 모델
optimizer = Adam(learning_rate = 0.001)
def my_model(opti = optimizer):
    input = Input(shape = (30, 30, 3))
    x = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(input)
    x = BatchNormalization()(x)
    x = Conv2D(64, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, 1, padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, 1, padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(64, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation = 'relu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = input, outputs = output)
    model.compile(loss = 'binary_crossentropy', optimizer = opti, metrics = ['acc'])

    return model

model = my_model()
model.fit(x_train, y_train, epochs = 100, batch_size = 10, validation_split = 0.2)

loss, acc = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('acc: ', acc)

# loss:  0.5601547360420227
# acc:  0.7400000095367432