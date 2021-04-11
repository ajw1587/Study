import numpy as np
import pandas as pd
import cv2 as cv
import tensorflow as tf
import cv2 as cv
import glob
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, train_test_split

# Data가 String일때 to_categorical 하는법
# 1. pandas.get_dummies(y_train)
#
# 2. from sklearn.preprocessing import LabelEncoder: String Data -> Number Data Change
#    code = np.array(code)
#    label_encoder = LabelEncoder()
#    vec = label_encoder.fit_transform(code)
#
# 3. integer_mapping = {x: i for i,x in enumerate(code)}
#    vec = [integer_mapping[word] for word in code]




# ######################### 1. Check Image Data Length
# img_path = glob.glob('F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/hangul-images/*.jpeg')
# print(len(img_path))
# # 67140ea Image Data

# ######################### 2. Read Image Data
# first_path = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/hangul-images/hangul_'
# last_path = '.jpeg'
# img_list = []
# for i in range(1, len(img_path) + 1):
#     img = cv.imread(first_path + str(i) + last_path, cv.IMREAD_GRAYSCALE)
#     img_list.append(img)
# x_train = np.array(img_list)
# print(x_train.shape)

# ######################### 3. Save x_train image data
# np.save('F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/ocr_x_train.npy', x_train)

#################################################################################################

x_train_path = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/ocr_x_train.npy'
y_train_path = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/labels-map.csv'

########################## Read Input Data
x_train = np.load(x_train_path)
print(x_train.shape)

########################## Read Label Data
y_train = pd.read_csv(y_train_path, header = None)
y_train = y_train.iloc[:, 1]

########################## String Label OneHotEncoding
onehot = LabelEncoder()
onehot_y_train = onehot.fit_transform(y_train)
# print(onehot_y_train.shape)
# result = np.where(onehot_y_train == 18)
# print(result)
# print(result[0][0])
# print(y_train[result[0][0]])

y_categorical = to_categorical(onehot_y_train.reshape(-1, 1))
print(y_categorical.shape)

########################## Model
eff04 = EfficientNetB4(include_top = False, weights = 'imagenet', input_shape = (64, 64, 1))
initial_model = eff04
last = eff04.output

x = Flatten()(last)
x = Dense(32)(x)
x = Dense(16)(x)
output1 = Dense(10, activation = 'softmax')(x)
model = Model(inputs = initial_model.input, outputs = output1)


# dense = Conv2D(256, 2, 1, padding = 'same', activation = 'relu')(input)
# dense = MaxPooling2D(pool_size = (2, 2))(dense)
# dense = BatchNormalization()(dense)
# dense = Dropout(0.3)(dense)

# dense = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(input)
# dense = MaxPooling2D(pool_size = (2, 2))(dense)
# dense = BatchNormalization()(dense)
# dense = Dropout(0.3)(dense)

# dense = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(input)
# dense = MaxPooling2D(pool_size = (2, 2))(dense)
# dense = BatchNormalization()(dense)
# dense = Dropout(0.3)(dense)

# flatten = Flatten()(dense)

hist = model.compile(optimizer = Adam(learning_rate = 0.001),
                     loss = 'categorical_crossentropy',
                     metrics = ['acc'])
model.fit(x_train, y_categorical, batch_size = 32, epochs = 100, validation_data = (x_val, y_val))


result = model.evaluate(x_test, y_test, batch_size = 256)
y_predict = model.predict(x_test)
