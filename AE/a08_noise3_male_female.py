# [실습] keras67_1 남자 여자에 noise 넣어서
# 기미, 주근깨, 여드름을 제거하시오.

# 실습
# 남자 여자 구별
# ImageDataGenerator, fit_generator 사용
# female: 895장, 300 x 300
# male: 841장, 300 x 300
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Conv2DTranspose, Conv2D, MaxPooling2D, Flatten, Dropout
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
    batch_size = 1736,
    class_mode = 'binary'
    # ,save_to_dir = '../data/image/gender_generator/train'
)

test_xy = test_aug.flow_from_directory(
    '../data/image/gender/train',
    target_size = (30, 30),
    batch_size = 300,
    class_mode = 'binary'
)

np.save('../data/image/gender/x_train.npy', arr = train_xy[0][0])
np.save('../data/image/gender/y_train.npy', arr = train_xy[0][1])
np.save('../data/image/gender/x_test.npy', arr = test_xy[0][0])
np.save('../data/image/gender/y_test.npy', arr = test_xy[0][1])
'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'auto')
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint/MaleFemale/noise_male_female.hdf5',
                     save_best_only = True, monitor = 'val_loss', mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 25, mode = 'auto')

# 1. 데이터
x_train = np.load('../data/image/gender/npy/keras67_2_train_x.npy')
x_test =  np.load('../data/image/gender/npy/keras67_2_test_x.npy')

# print(type(x_train))
# print(x_train.shape)        # (1736, 30, 30, 3)

x_train_noised = x_train + np.random.normal(0, 0.1, x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, x_test.shape)


# 2. 모델
optimizer = Adam(learning_rate = 0.001)
def my_model(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), 1,  input_shape = (30, 30, 3)))
    model.add(Conv2D(64, (3, 3), 1,  activation = 'relu'))
    model.add(Dense(hidden_layer_size, activation = 'relu'))
    model.add(Conv2DTranspose(64, (3, 3), 1,  activation = 'relu'))
    model.add(Conv2DTranspose(128, (3, 3), 1, activation = 'relu'))
    model.add(Dense(3, activation = 'sigmoid'))

    return model

model = my_model(154)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['mae'])
model.fit(x_train_noised, x_train, epochs = 500, validation_split = 0.2,
          callbacks = [cp, es, reduce_lr])


model2 = load_model('../data/modelcheckpoint/MaleFemale/noise_male_female.hdf5')
output = model2.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
       plt.subplots(3, 5, figsize = (20, 7))

# 이미지 다섯개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)
asd = np.array(random_images)
print(asd.shape)    # (5,)
print(asd[0])       # 9
print(asd[1])       # 177
print(asd[2])       # 176
print(asd[3])       # 69
print(asd[4])       # 53

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(30, 30, 3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(30, 30, 3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('NOISE', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(30, 30, 3), cmap = 'gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()