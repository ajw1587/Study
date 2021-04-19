import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Image 전처리
height = 800
width = 600

img = cv.imread('F:/Team Project/Image_data/ex01.jpg', cv.IMREAD_COLOR)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = cv.resize(img, dsize = (width, height))
img2 = cv.resize(img2,dsize =(width, height))

# DoG
gaussian1 = cv.GaussianBlur(img2, (3, 3), 1.6)
gaussian2 = cv.GaussianBlur(img2, (5, 5), 1)
img2 = gaussian2 - gaussian1

kernel = np.ones((2, 2), np.uint8)
img2 = cv.erode(img2, kernel, iterations = 1)
# img2 = cv.morphologyEx(img2, cv.MORPH_CLOSE, kernel)

# Canny
canny = cv.Canny(img, 80, 85)

print(img2)

cv.imshow('img', img)
cv.imshow('img2', img2)
cv.imshow('Canny', canny)
cv.waitKey(0)
cv.destroyAllWindows()

# # Histogram
# print(type(img2))
# print(img2)
# cv.imshow('img2', img2)
# cv.waitKey(0)
# cv.destroyAllWindows()