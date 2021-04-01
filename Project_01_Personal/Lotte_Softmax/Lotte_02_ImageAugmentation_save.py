import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train = np.load('../data/lotte/lotte_data/train_x(256,256).npy')
y_train = np.load('../data/lotte/lotte_data/train_y(256,256).npy')


image_gen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1
)

train_dataset = image_gen.flow(x_train,
                               y_train,
                               batch_size = 48000,
                               shuffle = True,
                               seed = 77)


np.save('../data/lotte/lotte_data/train_x(256,256)_aug.npy', arr = train_dataset[0][0])
np.save('../data/lotte/lotte_data/train_y(256,256)_aug.npy', arr = train_dataset[0][1])

# 이미지 자체를 저장하는 방법, categorical은 for문을 돌린다.
# i = 0
# path = '../data/lotte/0'
# for batch in image_gen.flow_from_directory('../data/lotte/train', 
#                                            target_size = (150, 150),
#                                            batch_size = 48,
#                                            class_mode = 'categorical',
#                                            save_to_dir = path, 
#                                            shuffle = False):
#     i += 1
#     print(i)
#     if i > 999:
#         break