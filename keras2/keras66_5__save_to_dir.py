import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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
    zoom_range = 0.3,
    shear_range = 0.7,
    fill_mode = 'nearest'
)
test_datagen = ImageDataGenerator(rescale = 1./255)

# flow 또는 flow_from_directory
# train_generator
xy_train = train_datagen.flow_from_directory(
    '../data/image/brain/train',
    target_size = (150, 150),
    batch_size = 2,                 # 가져올 이미지의 수
    class_mode = 'binary'            # 파일단위로 y값을 나뉜다.
    # , save_to_dir = '../data/image/brain_generator/train'
)

''' 생성된 이미지 확인하기
x_train, y_train = xy_train.next()
print(type(x_train))
print(type(y_train))
print(x_train.shape)
print(y_train.shape)
for idx in range(len(x_train)):  
    print(x_train[idx].shape)
    print(y_train[idx])
    plt.imshow(x_train[idx])
    plt.show()
'''

# test_generator
xy_test = test_datagen.flow_from_directory(
    '../data/image/brain/test',
    target_size = (150, 150),
    batch_size = 1,                 # 가져올 이미지의 수
    class_mode = 'binary'            # 파일단위로 y값을 나뉜다.
    , save_to_dir = '../data/image/brain_generator/test'
)
# print(xy_train[0][0].shape)           # Augmentation x값
print(xy_train[0][1].shape)           # Augmentation y값
print(xy_test[0][0].shape)
print(xy_train[0][1].shape)           # Augmentation y값
print(xy_test[0][0].shape)
# 설정후 flow를 건드릴때마다 이미지가 생성된다.
# xy_train, xy_test 총 2번씩 이미지 생성!
