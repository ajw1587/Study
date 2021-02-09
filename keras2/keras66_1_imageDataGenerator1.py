import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator & data augmentation
# idg = ImageDataGenerator(rescale=1./255,           # 리스케일링
#                          rotation_range = 20,      # 이미지 회전
#                          width_shift_range=0.3,    # 좌우 이동
#                          height_shift_range=0.3,   # 상하 이동
#                          shear_range=0.5,          # 밀림 강도
#                          zoom_range=0.2,           # 확대
#                          horizontal_flip=True,     # 좌우 반전
#                          vertical_flip=True)       # 상하 반전
#                          fill_mode = 'nearest'     # Null을 주변의 중간값과 비슷한것으로 채우겠다.

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
    batch_size = 160,               # 가져올 이미지의 수
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

# print(xy_train[0])
print(xy_train[0][0])           # X값
print(xy_train[0][0].shape)     # batchsize = 5 -> (5, 150, 150, 3)
print(xy_train[0][1])           # Y값
print(xy_train[0][1].shape)     # batchsize = 5 -> (5,)