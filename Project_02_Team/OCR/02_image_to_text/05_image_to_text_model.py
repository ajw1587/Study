# import os
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense, Input, BatchNormalization, GlobalMaxPool2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.backend import conv2d
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

X_TRAIN_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/hangul-images'
Y_TRAIN_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/labels-map.csv'
MODEL_SAVE_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/model_checkpoint/05_modelcheckpoint/'

y_train = pd.read_csv(Y_TRAIN_PATH,encoding="utf-8")

def Recognition_model():
    input = Input(shape=(64,64,3))
    x = Conv2D(filters=(64), kernel_size=(64,21),strides=1)(input)
    x = BatchNormalization()(x)
    x1 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(21,64),strides=1)(input)
    x = BatchNormalization()(x)
    x2 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(21,21),strides=1)(input)
    x = BatchNormalization()(x)
    x3 = GlobalMaxPool2D()(x)

    x = Conv2D(filters=(64), kernel_size=(5,5),strides=1)(input)
    x = BatchNormalization()(x)
    x4 = GlobalMaxPool2D()(x)

    x5 = Flatten()(x1+x2+x3+x4)

    x = Dense(1512)(x5)
    output = Dense(2220, activation="softmax")(x)

    model = Model(inputs=input, outputs= output)

    model.summary()
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['acc'])
    return model

model = Recognition_model()
y_train = shuffle(y_train)
# 759240
# 39960,42180,50616,


es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto')
rl = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 10)
cp = ModelCheckpoint(filepath = MODEL_SAVE_PATH, monitor = 'val_loss', save_best_only = True, mode = 'auto')
for i in range(0,15):
    step = 50616
    # step = 10
    lst_image_path = list(y_train.iloc[step*i:step*(i+1),0])
    lst_image_label = list(y_train.iloc[step*i:step*(i+1),-1])
    
    print(lst_image_path)
    print(type(lst_image_path))
    # lst_image_path = []
    # for j in range(step*i + 1, step*(i + 1) + 1):
    #     sub_path = X_TRAIN_PATH + 'hangul_' +str(j) + 'jpeg'
    #     lst_image_path.append(sub_path)
    # lst_image_label = list(y_train.iloc[step*i:step*(i+1),-1])
    # print(lst_image_path)
    # print(lst_image_label)
    # print('\n')
    # print('\n')

    x_images = []
    for i, image_path in enumerate(lst_image_path) :
        img = cv2.imread(image_path)
        x_images.append(img)
        print(i)

    x_images = np.array(x_images)
    y_label = np.array(lst_image_label)

    print(x_images)
    print(y_label)

    print(y_label.shape)
    print(x_images.shape)
    print(np.unique(y_label))
    
    # (37599,)
    # (37599, 64, 64, 3)

    y_label = y_label.reshape(-1,1)
    onehot = OneHotEncoder()
    onehot.fit(y_label)
    y_label = onehot.transform(y_label).toarray()

    print(y_label)
    print(y_label.shape)

    x_images = x_images/255.
    print(x_images.shape)

    kfold = KFold(n_splits=5, shuffle=True)
    a = 1
    for train_index, test_index in kfold.split(x_images):
        print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
        x_train, x_val = x_images[train_index], x_images[test_index]
        y_train, y_val = y_label[train_index], y_label[test_index]

        model.fit(x_train, y_train, batch_size = 100, epochs = 300, validation_data=(x_val, y_val), shuffle=True
                  , callbacks = [es, rl])
        if a == 5:
            model.save(MODEL_SAVE_PATH + str(i) + '.hdf5')
        a += 1
