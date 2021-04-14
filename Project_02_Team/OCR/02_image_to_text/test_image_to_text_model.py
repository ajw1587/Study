import numpy as np
import pandas as pd
import cv2 as cv
from tensorflow.keras.models import load_model

Y_TRAIN_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/labels-map.csv'
TEST_IMAGE_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/hangul-images/hangul_200.jpeg'
MODEL_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/model_checkpoint/03_image_to_text_mujin.hdf5'

# Test Image
img = cv.imread(TEST_IMAGE_PATH, cv.IMREAD_COLOR)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
img = img.reshape(1, img.shape[0], img.shape[1], 3)
print('1: ', type(img))
print('2: ', img.shape)

# Model
model = load_model(MODEL_PATH)
model.summary()
y_pred = model.predict(img)
print('3: ', np.argmax(y_pred))
print('4: ', y_pred)

# Test
onehot_y_train = np.load('F:/Team Project/OCR/02_Image_to_Text_model/onehot_y_train.npy')
print('5: ', onehot_y_train.shape)

result = np.where(onehot_y_train == np.argmax(y_pred))
print('6: ', result)

y_train = pd.read_csv(Y_TRAIN_PATH, header = None)
y_train = y_train.iloc[:, 1]
print('7: ', y_train[result[0][0]])