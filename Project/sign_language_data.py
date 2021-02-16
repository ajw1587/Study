import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# file name: 'hand(1. 숫자)_(2. 숫자)_bot_seg_(3. 숫자)_cropped.jpeg'
# 1. 숫자: 1~5
# 2. 숫자: 0~9
# 3. 숫자: 1~5
# 4. Image_Size = 400 x 400
# bot 5, 
# Train Data
# train_img = []
# y_data = []
# for i in range(10):                   # 0 ~ 9: 숫자
#     for j in range(1, 6):             # 1 ~ 5: 손 종류
#         for k in range(1, 6):         # 1 ~ 5: 손 위치
#             try:
#                 filename = 'hand' + str(j) + '_' + str(i) + '_bot_seg_' + str(k) + '_cropped.jpeg'
#                 subset = cv2.imread('../data/sign_image/sign_language/asl_dataset/' + str(i) + '/' + filename
#                                     )
#                 subset = cv2.resize(subset, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
#                 y_data.append(i)
#                 plt.imshow(subset)
#                 plt.show()
#             except Exception as e:
#                 print('hand' + str(j) + '_' + str(i) + '_bot_seg_' + str(k) + '_cropped.jpeg')
# # plt.

# 손상된 이미지 걸러주기
# https://stackoverflow.com/questions/62220855/tensorflow-removing-jfif
# 헤더에 'JFIF' 문자열이 없는 잘못된 이미지를 삭제.


file_path = '../data/sign_image/sign_language/asl_dataset/asl_dataset'

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rotation_range = 0.3
)
except_datagen = ImageDataGenerator(
    rescale = 1./255
)

train_dataset = train_datagen.flow_from_directory(
    file_path, target_size = (64, 64), class_mode = 'categorical',
    batch_size = 8, seed = 77)

# print(train_dataset[0][0].shape)
# print(train_dataset[0][1].shape)
plt.imshow(train_dataset[0][0][0])
plt.show()