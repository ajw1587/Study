import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode = 'nearest'
)
test_datagen = ImageDataGenerator(rescale = 1./255)

# flow 또는 flow_from_directory
# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size = (150, 150),
    batch_size = 5,                 # 가져올 이미지의 수
    class_mode = 'binary'           # 파일단위로 y값을 나뉜다.
)                                   # (80, 150, 150, 1)

# 한파일당 80개의 이미지, 총 두개의 파일
# 1번 파일의 y값은 0, 2번 파일의 y값은 1 
# batch_size = 10이면, 총 160개의 이미지를 10개의 단위로 잘라서 반환해준다.
# ex) batch_size = 10이면, xy_train[0] ~ xy_train[15]
# ex) batch_size = 5이면, xy_train[0] ~ xy_train[31]
# ex) batch_size = 160이면, xy_train[0] ~ X

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size = (150, 150),
    batch_size = 5,                 # 가져올 이미지의 수
    class_mode = 'binary'           # 파일단위로 y값을 나뉜다.
)

# print(xy_train)
# Found 160 images belonging to 2 classes.
# Found 120 images belonging to 2 classes.
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001B6D61E8550>

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3)))
model.add(Conv2D(64, 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, 2, activation = 'relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(64, 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, 2, activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.01), metrics = ['acc'])

history = model.fit_generator(
    xy_train, steps_per_epoch = 31, epochs = 80,
    validation_data = xy_test, validation_steps = 4
)
# steps_per_epoch = flow_from_directory에서 얼마나 많은 Data를 뽑아 사용할건지.
# ex) setps_per_epoch가 100이고 batch_size = 5면은 5개의 데이터를 100번 불러와 총 500개의 데이터가 fit
# xy_train = train_datagen.flow_from_directory(
#     '../data/image/brain/train',
#     target_size = (150, 150),
#     batch_size = 5,                 # 가져올 이미지의 수
#     class_mode = 'binary'           # 파일단위로 y값을 나뉜다.
# )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 시각화 할것!!!
import matplotlib.pyplot as plt

print('acc: ', acc[-1])
print('val_acc: ', val_acc[:-1])

plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.plot(range(80), acc, color = 'red')
plt.plot(range(80), val_acc, color = 'blue')
plt.legend(['ACC', 'VAL_ACC'])

plt.subplot(1, 2, 2)
plt.plot(range(80), loss, color = 'red')
plt.plot(range(80), val_loss, color = 'blue')
plt.legend(['LOSS', 'VAL_LOSS'])

plt.show()