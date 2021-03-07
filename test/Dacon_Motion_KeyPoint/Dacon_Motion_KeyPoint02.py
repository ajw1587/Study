# 데이콘야!
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2 as cv
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# 경로 설정
import glob
train_path = glob.glob('../data/image/Dacon_Motion_KeyPoint/1.open/train_imgs/*.jpg')
test_path = glob.glob('../data/image/Dacon_Motion_KeyPoint/1.open/test_imgs/*.jpg')

# 결과 데이터 불러오기
train_sub = pd.read_csv('../data/image/Dacon_Motion_KeyPoint/1.open/train_df.csv', index_col = 0)
submission = pd.read_csv('../data/image/Dacon_Motion_KeyPoint/1.open/sample_submission.csv', index_col = 0)
# print(train_sub.shape)      # (4195, 48)
# print(submission.shape)     # (1600, 48)


# # 시각화
# plt.figure(figsize = (8, 4))
# path = train_path[1]
# img = Image.open(path)

# keypoint = train_sub.iloc[1, :]

# for i in range(0, len(train_sub.columns), 2):
#     plt.plot(keypoint[i], keypoint[i +1], 'ro')
# plt.imshow(img)
# plt.show()


# Dataset 만들어주기
def trainGenerator():
    for i in range(len(train_path)):
        img = tf.io.read_file(train_path[i])
        img = tf.image.decode_jpeg(img, channels = 3)
        img = tf.image.resize(img, [180, 320])
        target = train_sub.iloc[i, :]
        yield(img, target)
train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32), (tf.TensorShape([180,320,3]), tf.TensorShape([48])))
train_dataset = train_dataset.batch(32).prefetch(1)

# from_generator(
#     generator, output_types, output_shapes=None, args=None
# )
# 첫번째 인자는 데이터를 제공해 줄 generator, 중요한 점은 호출하면 generator를 돌려주는 함수(또는 callable)여야한다는 점
# 두번째 인자는 generator가 돌려주는 데이터의 Type
# 세 번째 인자는 generator 돌려주는 데이터의 Shape
# 네 번째 인자는 generator에게 전달해 줄 인자들

# Model
efficient = EfficientNetB5(include_top = False, weights = 'imagenet', input_shape = (180, 320, 3))
efficient.trainable = False

model = Sequential()
model.add(efficient)
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(48))

cp_path = '../data/modelcheckpoint/Motion_KeyPoint/model/Dacon_Motion_KeyPoint02.hdf5'
es = EarlyStopping(monitor = 'loss', patience = 40, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 20, mode = 'auto')
cp = ModelCheckpoint(cp_path, monitor = 'loss', save_best_only = True, mode = 'auto')

model.compile(loss = 'mse', optimizer = Adam(learning_rate = 0.01), metrics = ['mse'])
model.fit(train_dataset, epochs = 200, verbose = 1, callbacks = [cp, reduce_lr, es])



# test Data
x_test = []

for test_path in tqdm(test_path):
    img = tf.io.read_file(test_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, [180, 320])
    x_test.append(img)

x_test = tf.stack(x_test, axis = 0)

model2 = load_model(cp_path)
pred = model2.predict(x_test)

submission.iloc[:, :] = pred

submission.to_csv('../data/modelcheckpoint/Motion_KeyPoint/submission/Dacon_Motion_KeyPoint2.csv')