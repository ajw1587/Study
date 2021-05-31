# https://herbwood.tistory.com/20
# mask rcnn

import numpy as np
import cv2 as cv

img = cv.imread('F:/Team Project/OCR/01_Text_detection/data/train_hangul-images/hangul_1.png')
print(img.shape)
cv.rectangle(img, (429, 285), (436, 292), (0, 255, 0), 1)
cv.rectangle(img, (398, 456), (408, 466), (0, 255, 0), 1)
cv.rectangle(img, (449, 278), (469, 298), (0, 255, 0), 1)

cv.imshow('img', img)
cv.waitKey(0)