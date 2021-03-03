# 데이콘야!
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from PIL import Image

# 경로 설정
import glob
train_path = glob.glob('../data/image/Dacon_Motion_KeyPoint/1.open/train_imgs/*.jpg')
test_path = glob.glob('../data/image/Dacon_Motion_KeyPoint/1.open/test_imgs/*.jpg')

# 결과 데이터 불러오기
train_sub = pd.read_csv('../data/image/Dacon_Motion_KeyPoint/1.open/train_df.csv', index_col = 0)
submission = pd.read_csv('../data/image/Dacon_Motion_KeyPoint/1.open/sample_submission.csv', index_col = 0)
# print(train_sub.shape)      # (4195, 48)
# print(submission.shape)     # (1600, 48)

# Dataset 만들어주기
def trainGenerator():
    for i in range(len(train_path)):
        img = tf.io.read_file(train_path[i])
        img = tf.image.decode_jpef(img, channels = 3)
        img = tf.image.resizee(img, [180, 320])
        target = train_sub.iloc[i, :]
        yield(img, target)
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([180,320,3]),tf.TensorShape([48])))
train_dataset = train_dataset.batch(32).prefetch(1)


# # 시각화
# plt.figure(figsize = (20, 10))
# path = sample_train_path[1]
# img = Image.open(path)

# keypoint = train_sub.iloc[1, :]

# for i in range(0, len(train_sub.columns), 2):
#     plt.plot(keypoint[i], keypoint[i +1], 'ro')
# plt.imshow(img)
# plt.show()

