import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('F:/Team Project/Image_data/01.jpg', cv.IMREAD_COLOR)
# img = np.where(img < 100, 255, 0)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img2 = cv.bitwise_not(img2)
img2 = img2/255.
# pixel값 줄여주기 = 겹치는 글자를 최소화 해주기 위해
kernel = np.ones((3, 3), np.uint8)
img2 = cv.erode(img2, kernel, iterations=1)
print(img2.shape)   # (2400, 1080)

cv.imshow('img', img2)
cv.waitKey(0)
cv.destroyAllWindows()


# Line Histogram
def Line_Histogram(gray_img):
    line_sum = gray_img.sum(axis=1)
    hist_label = np.arange(0, gray_img.shape[0])
    print(line_sum.shape)
    print(hist_label.shape)

    plt.figure(figsize=(10, 10))
    plt.barh(hist_label, line_sum)
    plt.show()

# Word Histogram
def Word_Histogram(Line_img):
    word_sum = Line_img.sum(axis=0)
    word_label = np.arange(0, Line_img.shape[1])
    print(np.max(word_sum))
    print(word_sum.shape)
    print(word_label.shape)

    plt.figure(figsize=(10, 5))
    plt.bar(word_label, word_sum)
    plt.show()

# Line Split
def Line_Split(gray_img):
    line_sum = gray_img.sum(axis=1)
    line_idx = []
    sign = False        # True: 추출중, False: 0인 지점
    for i in range(line_sum.shape[0]):
        if sign == False:
            if line_sum[i] == 0:
                continue
            line_idx.append(i)
            sign = True
        else:   # sign == True
            if line_sum[i] != 0:
                continue
            line_idx.append(i-1)
            sign = False

    line_img = []
    for k in range(0, len(line_idx), 2):
        line_img.append(img2[line_idx[k]: line_idx[k + 1], :])

    return line_img, line_idx

# Word Split
def Word_Split(line_img, line_idx): # line_img: numpy
    word_sum = line_img.sum(axis=0)
    word_idx = []
    print(type(line_idx))
    print(line_idx)
    sign = False        # True: 추출중, False: 0인 지점
    for i in range(word_sum.shape[0]):
        if sign == False:
            if word_sum[i] == 0:
                continue
            word_idx.append(i)
            sign = True
        else:   # sign == True
            if word_sum[i] != 0:
                continue
            word_idx.append(i)
            sign = False

    # '가' -> 'ㄱ' 'ㅏ'로 나오는 현상 없애주기
    print(word_idx)
    # [176, 240, 241, 307, 313, 369, 380, 422, 423, 434, 446, 499, 543, 602, 608, 671, 678, 679, 680, 731, 746, 779, 781, 808, 840, 904, 911, 961] 
    # (1, 2), (3, 4), (5, 6)... 중심값 거리 비교 (Center_Value)
    for i in range(0, len(word_idx), 2):
        first_CV = (word_idx[i] + word_idx[i + 1])/2

    word_img = []
    # print(word_idx)
    # print(len(word_idx))
    # for k in range(0, len(word_idx), 2):
    #     word_img.append(img[line_idx[0]: line_idx[1], word_idx[k]: word_idx[k + 1]])
    #     cv.imshow('image', img[line_idx[0] : line_idx[1] , word_idx[k]: word_idx[k + 1]])
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    return word_img

# 적용
line_img, line_idx = Line_Split(img2)
for i in range(len(line_img)):
    word_img = Word_Split(line_img[i], line_idx[i * 2 : i * 2 + 2])
