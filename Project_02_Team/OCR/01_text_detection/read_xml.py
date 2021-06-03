# https://herbwood.tistory.com/20
# https://github.com/Kanghee-Lee/Mask-RCNN_TF
# mask rcnn

# import numpy as np
# import cv2 as cv

# img = cv.imread('F:/Team Project/OCR/01_Text_detection/data/train_hangul-images/hangul_1.png')
# print(img.shape)
# cv.rectangle(img, (429, 285), (436, 292), (0, 255, 0), 1)
# cv.rectangle(img, (398, 456), (408, 466), (0, 255, 0), 1)
# cv.rectangle(img, (449, 278), (469, 298), (0, 255, 0), 1)

# cv.imshow('img', img)
# cv.waitKey(0)

from xml.etree.ElementTree import parse

tree = parse('F:/Team_Project/OCR/01_Text_detection/data/train_annotation/hangul_1.xml')
root = tree.getroot()

object_tag = root.findall('object')
# print(object_tag)
# print(len(object_tag))
# print(object_tag[0].find('bndbox').findtext('xmin'))
# print(object_tag[0].find('bndbox').find('xmin').text)

x1_min = object_tag[0].find('bndbox').findtext('xmin')
x1_max = object_tag[0].find('bndbox').findtext('xmax')
y1_min = object_tag[0].find('bndbox').findtext('ymin')
y1_max = object_tag[0].find('bndbox').findtext('ymax')

print(x1_min)
print(x1_max)
print(y1_min)
print(y1_max)