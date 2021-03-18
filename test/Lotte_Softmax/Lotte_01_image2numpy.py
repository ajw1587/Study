import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(
    rescale = 1./255,
)

train_dataset = image_gen.flow_from_directory(
    '../data/lotte/train',
    target_size = (64, 64),
    batch_size = 48000,
    class_mode = 'categorical'
)

print(train_dataset[0][0].shape)
print(type(train_dataset[0][0]))
print(train_dataset[0][1].shape)
print(type(train_dataset[0][1]))

image_arr = []
for i in range(72000):
    path = '../data/lotte/test/' + str(i) + '.jpg'
    image = cv.imread(path)
    image = cv.resize(image, (64, 64), interpolation = cv.INTER_CUBIC)
    image_arr.append(image)
    print(i)
image_arr = np.asarray(image_arr)

print(image_arr.shape)
print(type(image_arr))

np.save('../data/lotte/train_x(64,64).npy', arr = train_dataset[0][0])
np.save('../data/lotte/train_y(64,64).npy', arr = train_dataset[0][1])
np.save('../data/lotte/predict_x(64,64).npy', arr = image_arr)