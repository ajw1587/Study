import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1
)

train_dataset = image_gen.flow_from_directory(
    '../data/lotte/train',
    target_size = (150, 150),
    batch_size = 48000,
    class_mode = 'categorical',
    shuffle = True,
    seed = 77
    # save_to_dir = '../data/lotte/0'
)

print(train_dataset[0][0].shape)
print(type(train_dataset[0][0]))
print(train_dataset[0][1].shape)
print(type(train_dataset[0][1]))

np.save('../data/lotte/train_aug_x.npy', arr = train_dataset[0][0])
np.save('../data/lotte/train_aug_y.npy', arr = train_dataset[0][1])

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

